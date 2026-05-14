(ns eca.llm-providers.bedrock-test
  (:require
   [cheshire.core :as json]
   [clojure.test :refer [deftest is testing]]
   [eca.client-test-helpers :refer [with-client-proxied]]
   [eca.llm-providers.bedrock :as llm-providers.bedrock]
   [matcher-combinators.test :refer [match?]])
  (:import
   [java.io ByteArrayInputStream ByteArrayOutputStream]))

;; --- event-stream binary frame helpers ---

(defn ^:private write-u32! [^ByteArrayOutputStream baos n]
  (doseq [shift [24 16 8 0]]
    (.write baos (bit-and (bit-shift-right (long n) shift) 0xff))))

(defn ^:private string-header [name value]
  (let [name-bytes (.getBytes ^String name "UTF-8")
        value-bytes (.getBytes ^String value "UTF-8")
        baos (ByteArrayOutputStream.)]
    (.write baos (alength name-bytes))
    (.write baos name-bytes)
    (.write baos 7) ;; string header type
    (.write baos (bit-and (bit-shift-right (alength value-bytes) 8) 0xff))
    (.write baos (bit-and (alength value-bytes) 0xff))
    (.write baos value-bytes)
    (.toByteArray baos)))

(defn ^:private make-frame
  "Builds a single AWS event-stream frame. CRCs are written as zero since the
   decoder does not validate them."
  [event-type payload-map]
  (let [headers (string-header ":event-type" event-type)
        payload (.getBytes ^String (json/generate-string payload-map) "UTF-8")
        baos (ByteArrayOutputStream.)]
    (write-u32! baos (+ 12 (alength headers) (alength payload) 4))
    (write-u32! baos (alength headers))
    (write-u32! baos 0) ;; prelude crc
    (.write baos headers)
    (.write baos payload)
    (write-u32! baos 0) ;; message crc
    (.toByteArray baos)))

(deftest event-stream-seq-test
  (testing "decodes consecutive event-stream frames into [event-type data] pairs"
    (let [frames (byte-array (concat (make-frame "messageStart" {:role "assistant"})
                                     (make-frame "contentBlockDelta" {:contentBlockIndex 0
                                                                      :delta {:text "hi"}})
                                     (make-frame "messageStop" {:stopReason "end_turn"})))
          events (vec (#'llm-providers.bedrock/event-stream-seq (ByteArrayInputStream. frames)))]
      (is (= [["messageStart" {:role "assistant"}]
              ["contentBlockDelta" {:contentBlockIndex 0 :delta {:text "hi"}}]
              ["messageStop" {:stopReason "end_turn"}]]
             events))))
  (testing "returns empty seq on a clean EOF"
    (is (= [] (vec (#'llm-providers.bedrock/event-stream-seq (ByteArrayInputStream. (byte-array 0))))))))

(deftest normalize-conversation-test
  (testing "string content becomes a single text block"
    (is (match? [{:role "user" :content [{:text "hello"}]}]
                (#'llm-providers.bedrock/normalize-conversation
                 [] [{:role "user" :content "hello"}] false))))

  (testing "tool call + output become Converse toolUse / toolResult blocks"
    (is (match? [{:role "assistant"
                  :content [{:toolUse {:toolUseId "t1" :name "get_weather" :input {:city "Paris"}}}]}
                 {:role "user"
                  :content [{:toolResult {:toolUseId "t1"
                                          :status "success"
                                          :content [{:text string?}]}}]}]
                (#'llm-providers.bedrock/normalize-conversation
                 [{:role "tool_call" :content {:id "t1" :full-name "get_weather" :arguments {:city "Paris"}}}
                  {:role "tool_call_output" :content {:id "t1" :output {:contents [{:type :text :text "sunny"}]}}}]
                 nil false))))

  (testing "consecutive same-role messages are merged"
    (is (match? [{:role "user" :content [{:text "a"} {:text "b"}]}]
                (#'llm-providers.bedrock/normalize-conversation
                 [{:role "user" :content "a"}] [{:role "user" :content "b"}] false)))))

(deftest build-body-test
  (testing "filters extra-payload to Converse-valid keys and deep-merges inferenceConfig"
    (let [body (#'llm-providers.bedrock/build-body
                {:messages []
                 :instructions "sys"
                 :max-output-tokens 500
                 :extra-payload {:thinking {:type "adaptive"}
                                 :inferenceConfig {:temperature 0.5}}})]
      (is (nil? (:thinking body)))
      (is (match? {:inferenceConfig {:maxTokens 500 :temperature 0.5}
                   :system [{:text "sys"}]}
                  body))))

  (testing "reasoning adds reasoning_config to additionalModelRequestFields"
    (is (match? {:additionalModelRequestFields {:reasoning_config {:type "enabled"}}}
                (#'llm-providers.bedrock/build-body {:messages [] :reason? true})))))

(deftest converse-request-test
  (testing "constructs a Converse request and parses the response"
    (let [req* (atom nil)]
      (with-client-proxied {}
        (fn handler [req]
          (reset! req* req)
          {:status 200
           :body {:output {:message {:role "assistant" :content [{:text "Hello!"}]}}
                  :stopReason "end_turn"
                  :usage {:inputTokens 10 :outputTokens 3}}})

        (let [result (llm-providers.bedrock/chat!
                      {:model "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
                       :api-url "http://localhost:1"
                       :api-key "fake-token"
                       :instructions "be terse"
                       :user-messages [{:role "user" :content [{:type :text :text "hi"}]}]
                       :past-messages []
                       :max-output-tokens 1000}
                      nil)]
          (is (match? {:method "POST"
                       :uri "/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/converse"}
                      (select-keys @req* [:method :uri])))
          (is (= "Bearer fake-token" (get-in @req* [:headers "Authorization"])))
          (is (match? {:messages [{:role "user" :content [{:text "hi"}]}]
                       :system [{:text "be terse"}]
                       :inferenceConfig {:maxTokens 1000}}
                      (:body @req*)))
          (is (match? {:output-text "Hello!"
                       :usage {:input-tokens 10 :output-tokens 3}}
                      result)))))))
