# 第169-170页shell命令----------------------------------------------------------------
pip install transformers

pip install transformers[tf-cpu]
# pip install transformers[tf-gpu]

conda install -c huggingface transformers

git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

pip install --upgrade tensorflow-cpu


# 第170-171页，第7.2.2小节代码----------------------------------------------------------------
from transformers import pipeline
classifier = pipeline('sentiment-analysis')

classifier('We are very happy to show you the 🤗 Transformers library.')

results = classifier(["We are very happy to show you the 🤗 Transformers library.",
           "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
# This model only exists in PyTorch, so we use the `from_pt` flag to import that model in TensorFlow.
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


# 第172-173页，第7.2.3.1小节代码----------------------------------------------------------------
from transformers import TFBertForSequenceClassification
# 根据任务类别选择合适的模型类，此处选择句子分类任务的模型：TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

from transformers import BertTokenizer, glue_convert_examples_to_features
import tensorflow as tf
import tensorflow_datasets as tfds
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = tfds.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_dataset, epochs=2, steps_per_epoch=115)
# 训练完毕后保存模型
model.save_pretrained('./my_mrpc_model/')


# 第173-174页，第7.2.3.2小节代码----------------------------------------------------------------
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments

model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased")

training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tfds_train_dataset,    # tensorflow_datasets training dataset
    eval_dataset=tfds_test_dataset       # tensorflow_datasets evaluation dataset
)

# 微调模型
trainer.train()
# 验证模型
trainer.evaluate()


# 第175页，第7.2.4.1小节代码----------------------------------------------------------------
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
# 根据任务类型使用TFBertForSequenceClassification读取BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-cased')
# 构造输入文本和标签
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1)) # Batch size 1
# 利用模型推理得到预测结果
outputs = model(inputs)
loss = outputs.loss
logits = outputs.logits


# 第176页，第7.2.4.2小节代码----------------------------------------------------------------
from transformers import BertTokenizer, TFBertForMultipleChoice
import tensorflow as tf
# 根据任务类型使用TFBertForMultipleChoice读取BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForMultipleChoice.from_pretrained('bert-base-cased')
# 构造输入文本样例
prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
# 将输入文本改写成一一对应的句对字典
encoding = tokenizer([[prompt, prompt], [choice0, choice1]], return_tensors='tf', padding=True)
inputs = {k: tf.expand_dims(v, 0) for k, v in encoding.items()}
outputs = model(inputs)  # batch size is 1

# the linear classifier still needs to be trained
logits = outputs.logits


# 第177页，第7.2.4.3小节代码----------------------------------------------------------------
from transformers import BertTokenizer, TFBertForTokenClassification
import tensorflow as tf
# 根据任务类型使用TFBertForTokenClassification读取BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForTokenClassification.from_pretrained('bert-base-cased')
# 构造输入样本和标签
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
input_ids = inputs["input_ids"]
inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))) # Batch size 1

outputs = model(inputs)
loss = outputs.loss
logits = outputs.logits


# 第178页，第7.2.4.4小节代码----------------------------------------------------------------
from transformers import BertTokenizer, TFBertForQuestionAnswering
import tensorflow as tf
# 根据任务类型使用TFBertForTokenClassification读取BERT模型
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertForQuestionAnswering.from_pretrained('bert-base-cased')
# 构造输入样本
question, text = "Who was Jim Henson?", "Jim Henson was a cute robot"
input_dict = tokenizer(question, text, return_tensors='tf')
outputs = model(input_dict)
# 限定答案在答句中寻找
answer_field = input_dict.token_type_ids.numpy()
start_logits = outputs.start_logits + answer_field
end_logits = outputs.end_logits + answer_field
# 根据找到的答案起始位置和终止位置，输出答案
all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])






















