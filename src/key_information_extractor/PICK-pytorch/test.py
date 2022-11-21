# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import sys
sys.path = ["../../utils", "../../"] + sys.path
from utils import visualize_img
del sys.modules["utils"]
sys.path = sys.path[2:]
import argparse
from unittest import result
import torch
from tqdm import tqdm
from pathlib import Path
import os
from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import InvalidTagSequence

import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
import numpy as np

def bio_tags_to_spans(tag_sequences, text_length):
    text_index = [0]
    for length in text_length:
        text_index.append(text_index[-1] + length)
    spans = []
    list_idx = []
    for i, item in enumerate(tag_sequences):
        bio_tag = item[0]
        if (bio_tag not in ["B", "I", "O"]):
            raise InvalidTagSequence(tag_sequences)
        type_tag = item[2:]
        if (bio_tag == "B"):
            if i in text_index:
                spans.append((type_tag, (i, text_index[text_index.index(i) + 1] - 1)))
                list_idx.append(text_index.index(i))
    return spans, list_idx

def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    # print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    resized_image_size = (560, 784)    # w, h
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=resized_image_size,
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            # For easier debug.
            image_names = input_data_item["filenames"]

            output = pick_model(**input_data_item)
            logits = output['logits']  # (B, N*T, out_dim)
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            text_length = input_data_item['text_length'].cpu().numpy()[0]
            boxes_coors = input_data_item['boxes_coordinate'].cpu().numpy()[0]  # (num_boxes, 8)
            org_shapes = input_data_item['org_shapes'].cpu().numpy()[0]
            boxes_coors = np.rint(boxes_coors * np.tile(org_shapes/np.array(resized_image_size, dtype=np.int32), 4)).astype(np.int32)
            boxes_coors = boxes_coors[:, [2, 3, 4, 5, 6, 7, 0, 1]]
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                spans, list_idx = bio_tags_to_spans(decoded_tags, text_length)

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w', encoding='utf-8') as f:
                    for item, index in zip(entities, list_idx):
                        box_coors = boxes_coors[index]
                        boxes_coors_text = ",".join([str(item) for item in box_coors])
                        f.write('{},{},{}\n'.format(boxes_coors_text, item['text'], item['entity_name']))
                image_ext = os.path.splitext(image_names[0])[1]
                img_name = os.path.basename(result_file).replace(".txt", image_ext)
                visualize_img(os.path.join(args.impt, img_name), result_file, save_path=os.path.join(args.visualize_dir, img_name), kie_labels=True)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('--visualize_dir', default=None, type=str,
                      help='output visualize dir images folder path')
    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()
    main(args)
