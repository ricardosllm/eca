(ns eca.llm-providers.bedrock
  "AWS Bedrock provider using the native Converse / ConverseStream APIs.

   Bedrock exposes an OpenAI-compatible surface too, but it only serves the
   `openai.*` models — Claude (and other) inference profiles are reachable
   only through Converse, hence this dedicated handler.

   Auth is the bearer-token flow (`Authorization: Bearer <token>`), which
   sidesteps SigV4 and is the simplest path for a local desktop client."
  (:require
   [cheshire.core :as json]
   [clojure.string :as string]
   [eca.client-http :as client]
   [eca.llm-util :as llm-util]
   [eca.logger :as logger]
   [eca.shared :as shared :refer [assoc-some join-api-url]]
   [hato.client :as http])
  (:import
   [java.io InputStream]
   [java.net URLEncoder]))

(set! *warn-on-reflection* true)

(def ^:private logger-tag "[BEDROCK]")

(def ^:private default-max-output-tokens 32000)
(def ^:private default-reasoning-budget-tokens 2048)

;; --- Message normalization (ECA <-> Bedrock Converse blocks) ---

(defn ^:private image-format
  "Maps an `image/*` media type to the format string Bedrock expects."
  [media-type]
  (case media-type
    "image/jpeg" "jpeg"
    "image/gif" "gif"
    "image/webp" "webp"
    "png"))

(defn ^:private ->image-block [{:keys [media-type base64]}]
  {:image {:format (image-format media-type)
           :source {:bytes base64}}})

(defn ^:private ->tool-config
  "Builds the Converse `toolConfig` value from ECA tools, or nil when there
   are none."
  [tools]
  (when (seq tools)
    {:tools (mapv (fn [tool]
                    {:toolSpec {:name (:full-name tool)
                                :description (:description tool)
                                :inputSchema {:json (:parameters tool)}}})
                  tools)}))

(defn ^:private normalize-messages
  "Converts ECA's internal message format into Converse `{:role :content}`
   maps where content is always a vector of Converse content blocks."
  [messages supports-image?]
  (keep
   (fn [{:keys [role content] :as msg}]
     (case role
       "tool_call"
       {:role "assistant"
        :content [{:toolUse {:toolUseId (:id content)
                             :name (:full-name content)
                             :input (or (:arguments content) {})}}]}

       "tool_call_output"
       (let [contents (-> content :output :contents)
             image-contents (when supports-image?
                              (seq (filter #(= :image (:type %)) contents)))
             text (llm-util/stringfy-tool-result content)]
         {:role "user"
          :content [{:toolResult
                     {:toolUseId (:id content)
                      :content (into [{:text (if (string/blank? text) "(no content)" text)}]
                                     (map ->image-block)
                                     image-contents)
                      :status (if (:error content) "error" "success")}}]})

       "reason"
       {:role "assistant"
        :content [(if (:redacted? content)
                    {:reasoningContent {:redactedContent (:data content)}}
                    {:reasoningContent {:reasoningText (assoc-some {:text (:text content)}
                                                                  :signature (:external-id content))}})]}

       ;; "user" / "assistant"
       {:role role
        :content (if (string? content)
                   [{:text (string/trim content)}]
                   (vec (keep (fn [block]
                                (case (some-> (:type block) name)
                                  "text" {:text (:text block)}
                                  "image" (when supports-image? (->image-block block))
                                  nil))
                              content)))}))
   messages))

(defn ^:private group-parallel-tool-calls
  "Reorders consecutive tool_call/tool_call_output messages so all tool_calls
   precede all tool_call_outputs, keeping parallel tool turns grouped."
  [messages]
  (let [tool-msg? #(contains? #{"tool_call" "tool_call_output"} (:role %))]
    (->> messages
         (partition-by tool-msg?)
         (mapcat (fn [group]
                   (if-not (tool-msg? (first group))
                     group
                     (let [{calls "tool_call" outputs "tool_call_output"} (group-by :role group)
                           call-id->pos (into {} (map-indexed (fn [i m] [(get-in m [:content :id]) i]) calls))]
                       (concat calls (sort-by #(call-id->pos (get-in % [:content :id])) outputs)))))))))

(defn ^:private merge-adjacent-by-role
  "Merges consecutive messages of `role` into one, concatenating their content
   vectors. Converse rejects consecutive same-role messages."
  [role messages]
  (reduce
   (fn [acc msg]
     (let [prev (peek acc)]
       (if (and (= role (:role prev)) (= role (:role msg)))
         (conj (pop acc) {:role role :content (into (:content prev) (:content msg))})
         (conj acc msg))))
   []
   messages))

(defn ^:private normalize-conversation [past-messages user-messages supports-image?]
  (->> (concat past-messages user-messages)
       group-parallel-tool-calls
       (#(normalize-messages % supports-image?))
       (merge-adjacent-by-role "assistant")
       (merge-adjacent-by-role "user")
       vec))

;; --- Request body ---

(def ^:private allowed-extra-payload-keys
  "Top-level members the Converse API accepts. extraPayload is filtered to
   these so provider/model variant payloads aimed at other APIs (e.g.
   Anthropic's `:thinking`) don't reach Bedrock and trigger a 400."
  #{:additionalModelRequestFields :additionalModelResponseFieldPaths
    :guardrailConfig :inferenceConfig :performanceConfig
    :promptVariables :requestMetadata :system :toolConfig})

(defn ^:private build-body
  [{:keys [messages instructions max-output-tokens tools reason? extra-payload]}]
  (shared/deep-merge
   (assoc-some
    {:messages messages
     :inferenceConfig {:maxTokens (or max-output-tokens default-max-output-tokens)}}
    :system (when-not (string/blank? instructions) [{:text instructions}])
    :toolConfig (->tool-config tools)
    :additionalModelRequestFields (when reason?
                                    {:reasoning_config {:type "enabled"
                                                        :budget_tokens default-reasoning-budget-tokens}}))
   (select-keys extra-payload allowed-extra-payload-keys)))

;; --- AWS event-stream (vnd.amazon.eventstream) binary decoder ---

(defn ^:private u32
  "Reads a big-endian unsigned 32-bit integer from `b` at `off`."
  ^long [^bytes b ^long off]
  (bit-or (bit-shift-left (bit-and (long (aget b off)) 0xff) 24)
          (bit-shift-left (bit-and (long (aget b (+ off 1))) 0xff) 16)
          (bit-shift-left (bit-and (long (aget b (+ off 2))) 0xff) 8)
          (bit-and (long (aget b (+ off 3))) 0xff)))

(defn ^:private read-fully
  "Reads exactly `n` bytes from `is`. Returns the byte-array, nil on a clean
   EOF before any byte was read, or throws on a truncated frame."
  ^bytes [^InputStream is ^long n]
  (let [buf (byte-array n)
        first-read (.read is buf 0 n)]
    (cond
      (neg? first-read) nil
      :else (loop [off (long first-read)]
              (if (< off n)
                (let [r (.read is buf off (- n off))]
                  (if (neg? r)
                    (throw (ex-info "Unexpected EOF in Bedrock event-stream frame" {}))
                    (recur (+ off (long r)))))
                buf)))))

(defn ^:private parse-headers
  "Parses event-stream frame headers. Only the string header type (7) is
   handled — every header Bedrock emits for Converse stream events is a
   string."
  [^bytes b]
  (loop [pos 0
         acc {}]
    (if (>= pos (alength b))
      acc
      (let [name-len (bit-and (long (aget b pos)) 0xff)
            header-name (String. b (int (inc pos)) (int name-len) "UTF-8")
            type-pos (+ pos 1 name-len)
            header-type (bit-and (long (aget b type-pos)) 0xff)]
        (if (= 7 header-type)
          (let [value-len (bit-or (bit-shift-left (bit-and (long (aget b (+ type-pos 1))) 0xff) 8)
                                  (bit-and (long (aget b (+ type-pos 2))) 0xff))
                value-start (+ type-pos 3)]
            (recur (+ value-start value-len)
                   (assoc acc header-name (String. b (int value-start) (int value-len) "UTF-8"))))
          ;; Unknown header type: stop defensively, we already have what we need.
          acc)))))

(defn ^:private event-stream-seq
  "Lazily decodes an AWS event-stream into `[event-type data]` pairs.
   `event-type` falls back to the exception/message type header for error
   frames. Prelude and message CRCs are not validated."
  [^InputStream is]
  (lazy-seq
   (when-let [prelude (read-fully is 12)]
     (let [total-len (u32 prelude 0)
           headers-len (u32 prelude 4)
           payload-len (- total-len headers-len 16)
           headers (parse-headers (read-fully is headers-len))
           payload (read-fully is payload-len)
           _crc (read-fully is 4)
           event-type (or (get headers ":event-type")
                          (get headers ":exception-type")
                          (get headers ":message-type"))
           data (json/parse-string (String. ^bytes payload "UTF-8") true)]
       (cons [event-type data] (event-stream-seq is))))))

;; --- HTTP request (shared by stream / non-stream) ---

(declare ^:private response->result)

(defn ^:private base-request!
  [{:keys [rid body model api-url api-key extra-headers http-client cancelled?
           content-block* on-error on-stream on-tools-called-wrapper]}]
  (let [path (str "/model/" (URLEncoder/encode ^String model "UTF-8")
                  (if on-stream "/converse-stream" "/converse"))
        url (join-api-url api-url path)
        headers (client/merge-llm-headers
                 (merge {"Authorization" (str "Bearer " api-key)
                         "Content-Type" "application/json"}
                        extra-headers))
        response* (atom nil)
        on-error (if on-stream
                   on-error
                   (fn [error-data]
                     (llm-util/log-response logger-tag rid "response-error" error-data)
                     (reset! response* {:error error-data})))]
    (llm-util/log-request logger-tag rid url body headers)
    (try
      (let [{:keys [status body]} (http/post url
                                             {:headers headers
                                              :body (json/generate-string body)
                                              :throw-exceptions? false
                                              :http-client (client/merge-with-global-http-client http-client)
                                              :as (if on-stream :stream :json)})]
        (if (not= 200 status)
          (let [body-str (if on-stream (slurp body) (json/generate-string body))]
            (logger/warn logger-tag (format "[%s] Unexpected response status: %s body: %s" rid status body-str))
            (on-error {:message (format "Bedrock response status: %s body: %s" status body-str)
                       :status status
                       :body body-str}))
          (if on-stream
            (let [{:keys [touch-fn set-reading-fn stop-fn reason*]}
                  (llm-util/start-stream-watchdog! body cancelled? {})
                  completed?* (atom false)]
              (try
                (with-open [^InputStream is body]
                  (doseq [[event data] (event-stream-seq is)]
                    (set-reading-fn false)
                    (touch-fn)
                    (llm-util/log-response logger-tag rid event data)
                    (on-stream event data content-block*)
                    (when (= "messageStop" event)
                      (reset! completed?* true))
                    (set-reading-fn true)))
                (when-not (or @completed?* (cancelled?))
                  (logger/warn logger-tag "Stream ended without messageStop, retrying")
                  (on-error {:message "Stream ended without completion signal"
                             :error/type :premature-stop}))
                (catch java.io.IOException e
                  (case @reason*
                    :cancelled (throw (ex-info "Stream cancelled" {:silent? true}))
                    :idle-timeout (on-error {:message "Stream idle timeout: no data received" :exception e})
                    (on-error {:exception e :message (llm-util/connection-error-message e)})))
                (finally
                  (stop-fn))))
            (do
              (llm-util/log-response logger-tag rid "response" body)
              (reset! response* (response->result body on-tools-called-wrapper))))))
      (catch Exception e
        (on-error {:exception e :message (llm-util/connection-error-message e)})))
    @response*))

;; --- Non-streaming response ---

(defn ^:private parse-usage [usage]
  {:input-tokens (or (:inputTokens usage) 0)
   :output-tokens (or (:outputTokens usage) 0)
   :input-cache-read-tokens (or (:cacheReadInputTokens usage) 0)
   :input-cache-creation-tokens (or (:cacheWriteInputTokens usage) 0)})

(defn ^:private response->result [body on-tools-called-wrapper]
  (let [content (-> body :output :message :content)
        reason-block (some :reasoningContent content)
        tools-to-call (->> content
                           (keep :toolUse)
                           (mapv (fn [{:keys [toolUseId name input]}]
                                   {:id toolUseId
                                    :full-name name
                                    :arguments (or input {})})))]
    (assoc-some
     {:output-text (string/join (keep :text content))
      :usage (parse-usage (:usage body))}
     :reason-text (get-in reason-block [:reasoningText :text])
     :reason-id (when reason-block (str (random-uuid)))
     :tools-to-call (not-empty tools-to-call)
     :call-tools-fn (when (seq tools-to-call)
                      (fn [on-tools-called]
                        (on-tools-called-wrapper tools-to-call on-tools-called))))))

;; --- Entry point ---

(defn chat!
  [{:keys [model user-messages instructions max-output-tokens api-url api-key
           reason? past-messages tools extra-payload extra-headers supports-image?
           http-client cancelled?]}
   {:keys [on-message-received on-error on-reason on-prepare-tool-call
           on-tools-called on-usage-updated] :as callbacks}]
  (let [stream? (boolean callbacks)
        cancelled? (or cancelled? (constantly false))
        messages (normalize-conversation past-messages user-messages supports-image?)
        body (build-body {:messages messages
                          :instructions instructions
                          :max-output-tokens max-output-tokens
                          :tools tools
                          :reason? reason?
                          :extra-payload extra-payload})
        base-opts {:model model
                   :api-url api-url
                   :extra-headers extra-headers
                   :http-client http-client
                   :cancelled? cancelled?}
        ;; Re-issue the request after tools ran, rebuilding messages from the
        ;; updated history the chat pipeline hands back.
        on-tools-called-wrapper
        (fn on-tools-called-wrapper [tools-to-call on-tools-called]
          (when-let [{:keys [new-messages tools fresh-api-key]} (on-tools-called tools-to-call)]
            (let [messages (normalize-conversation new-messages nil supports-image?)]
              (base-request! (merge base-opts
                                    {:rid (llm-util/gen-rid)
                                     :body (assoc-some (assoc body :messages messages)
                                                       :toolConfig (->tool-config tools))
                                     :api-key (or fresh-api-key api-key)
                                     :on-tools-called-wrapper on-tools-called-wrapper})))))
        handle-stream
        (when stream?
          (fn handle-stream [event data content-block*]
            (case event
              "messageStart" nil

              "contentBlockStart"
              (when-let [tool-use (-> data :start :toolUse)]
                (swap! content-block* assoc (:contentBlockIndex data)
                       {:type :tool-use
                        :toolUseId (:toolUseId tool-use)
                        :name (:name tool-use)
                        :input-json ""})
                (on-prepare-tool-call {:full-name (:name tool-use)
                                       :id (:toolUseId tool-use)
                                       :arguments-text ""}))

              "contentBlockDelta"
              (let [idx (:contentBlockIndex data)
                    delta (:delta data)]
                (cond
                  (:text delta)
                  (on-message-received {:type :text :text (:text delta)})

                  (:toolUse delta)
                  (let [chunk (-> delta :toolUse :input)]
                    (swap! content-block* update-in [idx :input-json] str chunk)
                    (let [block (get @content-block* idx)]
                      (on-prepare-tool-call {:full-name (:name block)
                                             :id (:toolUseId block)
                                             :arguments-text (or chunk "")})))

                  (:reasoningContent delta)
                  (let [reasoning (:reasoningContent delta)
                        reason-id (or (get-in @content-block* [idx :reason-id])
                                      (let [new-id (str (random-uuid))]
                                        (swap! content-block* assoc idx {:type :reasoning :reason-id new-id})
                                        (on-reason {:status :started :id new-id})
                                        new-id))]
                    (cond
                      (:text reasoning)
                      (on-reason {:status :thinking :id reason-id :text (:text reasoning)})

                      (:signature reasoning)
                      (do (on-reason {:status :finished :id reason-id :external-id (:signature reasoning)})
                          (swap! content-block* assoc-in [idx :signature-done?] true))))))

              "contentBlockStop"
              (let [block (get @content-block* (:contentBlockIndex data))]
                (when (and (= :reasoning (:type block))
                           (not (:signature-done? block)))
                  (on-reason {:status :finished :id (:reason-id block)})))

              "messageStop"
              (case (:stopReason data)
                "tool_use"
                (let [tool-calls (->> (vals @content-block*)
                                      (filter #(= :tool-use (:type %)))
                                      (mapv (fn [{:keys [toolUseId name input-json]}]
                                              {:id toolUseId
                                               :full-name name
                                               :arguments (json/parse-string input-json)})))]
                  (when-let [{:keys [new-messages tools fresh-api-key]} (on-tools-called tool-calls)]
                    (base-request! (merge base-opts
                                          {:rid (llm-util/gen-rid)
                                           :body (assoc-some (assoc body :messages (normalize-conversation new-messages nil supports-image?))
                                                             :toolConfig (->tool-config tools))
                                           :api-key (or fresh-api-key api-key)
                                           :content-block* (atom {})
                                           :on-error on-error
                                           :on-stream handle-stream}))))

                "max_tokens"
                (on-message-received {:type :limit-reached})

                ;; "end_turn" / "stop_sequence" / "content_filtered" / "guardrail_intervened"
                (on-message-received {:type :finish :finish-reason (:stopReason data)}))

              "metadata"
              (when-let [usage (:usage data)]
                (on-usage-updated (parse-usage usage)))

              ;; exception / error frames
              (on-error {:message (format "Bedrock stream error (%s): %s" event data)}))))]
    (base-request! (merge base-opts
                          {:rid (llm-util/gen-rid)
                           :body body
                           :api-key api-key
                           :content-block* (atom {})
                           :on-error (or on-error identity)
                           :on-tools-called-wrapper on-tools-called-wrapper
                           :on-stream handle-stream}))))
