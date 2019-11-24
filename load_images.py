import argparse
import csv
import os
import time

import requests
import shutil
import _utils


def download_images(domain, images_csv, out_path="img"):
    with open(images_csv) as f:
        reader = csv.reader(f)
        row_count = sum(1 for _ in reader)

    _utils.print_progress(0, row_count, prefix='Progress:', suffix='Complete', bar_length=10)

    with open(images_csv) as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        for line in reader:
            f_url = line[1]
            _, file_ext = os.path.splitext(f_url)

            r = requests.get(domain + "/" + f_url, stream=True)
            if r.status_code == 200:
                with open(out_path + "/" + line[0] + file_ext, 'wb') as out_f:
                    r.raw.decode_content = True
                    shutil.copyfileobj(r.raw, out_f)
            else:
                print(f_url + "\nResponse code:", r.status_code)

            r.close()
            i += 1
            _utils.print_progress(i, row_count, prefix='Progress:', suffix=f_url, bar_length=10)
            time.sleep(.5)


def load_images(path, images_csv, out_path="img"):
    with open(images_csv) as f:
        reader = csv.reader(f)
        row_count = sum(1 for _ in reader)

    _utils.print_progress(0, row_count, prefix='Progress:', suffix='Complete', bar_length=40)

    with open(images_csv) as f:
        reader = csv.reader(f, delimiter=',')

        i = 0
        for line in reader:
            if not line[1]:
                continue

            src = path + "/" + line[1]
            src = src.replace("//", "/")
            _, file_ext = os.path.splitext(src)

            dist = out_path + "/" + line[0] + file_ext
            dist = dist.replace("//", "/")

            shutil.copyfile(src, dist)

            i += 1
            _utils.print_progress(i, row_count, prefix='Progress:', suffix="Complete", bar_length=40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Загрузка файлов для анализа.')
    parser.add_argument('-t', '--type', help='Источник. 0 - локальная папка, 1 - сайт в интернет',
                        required=True, type=int, choices=(0, 1))
    parser.add_argument('-s', '--source', help='Локальный путь или домен с протоколом.',
                        required=True, type=str)
    parser.add_argument('-l', '--list', help='CSV файл с адресами изображений. Формат: id, url',
                        default="files.csv", type=str)
    parser.add_argument('-o', '--outDir',
                        help='Папка для сохранения результатов',
                        default="img", type=str)

    args = parser.parse_args()
    Type = args.type
    Source = args.source
    CSVList = args.list
    outDir = args.outDir

    if Type == 0:
        load_images(Source, CSVList, outDir)
    else:
        download_images(Source, CSVList, outDir)
