import sys
sys.path = ["./vietocr_github"] + sys.path
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_seq2seq')

dataset_params = {
    'name':'th',
    'data_root':'../../dataset/text_recognition_mcocr_data',
    'train_annotation':'../../dataset/text_recognition_train_data.txt',
    'valid_annotation':'../../dataset/text_recognition_val_data.txt'
}

trainer_params = {
     'print_every':200,
     'valid_every':166,
      'iters':16600,
      'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
      'export':'./weights/transformerocr.pth',
      'metrics': 10000
}

optimizer_params = {
    'max_lr': 0.0001
}

config["dataset"].update(dataset_params)
config["trainer"].update(trainer_params)
config["optimizer"].update(optimizer_params)

print(config)

trainer = Trainer(config, pretrained=True)
trainer.visualize_dataset()
trainer.train()
trainer.visualize_prediction()
print("acc full sequence, acc per character: ", trainer.precision())