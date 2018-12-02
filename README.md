本模型是基于google bert Chinese Based pre-trained model 进行fine-tuning，进行的中文命名实体识别实验

训练模型：
python cmh_bert_ner.py

我所使用的是google Chinese Based预训练模型进行fine-tuning的，在google官网下载；根据需要，自行更改模型的相关训练参数

测试效果：
python evaluate.py

本人所使用数据是基于旅游游记的数据，模型指标如下：

eval_f = 0.84716296

eval_precision = 0.8513407

eval_recall = 0.84808767

global_step = 31140

loss = 0.09112783

参考：英文实体识别： https://github.com/kyzhouhzau/BERT-NER
