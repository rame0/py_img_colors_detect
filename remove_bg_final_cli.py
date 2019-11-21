import argparse, cv2, os
import numpy as np
import _utils


def remove_bg(image_name, in_path="img", out_path="out", main_rect_size=.02, fg_size=4, resize_to=500):
    img = cv2.imread(in_path + '/' + image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # quantify colors
    # img = _utils.quantify_colors(img, 32, 10)

    img_height, img_width = img.shape[:2]

    if img_width > img_height:
        height = resize_to
        width = round(img_width * height / img_height)
    else:
        width = resize_to
        height = round(img_height * width / img_width)

    # resize image to lower resources usage
    img_small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    # create mask tpl
    mask = np.zeros(img_small.shape[:2], np.uint8)

    # create BG rect
    bg_w = round(width * main_rect_size)
    bg_h = round(height * main_rect_size)
    bg_rect = (bg_w, bg_h, width - bg_w, height - bg_h)

    # create FG rect
    fg_w = round(width * (1 - fg_size) / 2)
    fg_h = round(height * (1 - fg_size) / 2)
    fg_rect = (fg_w, fg_h, width - fg_w, height - fg_h)

    # color: 0 - bg, 1 - fg, 2 - probable bg, 3 - probable fg
    cv2.rectangle(mask, fg_rect[:2], fg_rect[2:4], color=2, thickness=-1)
    # cv2.rectangle(mask, bg_rect[:2], bg_rect[2:4], color=1, thickness=bg_w*3)

    mask_preset = mask.copy()

    bgd_model1 = np.zeros((1, 65), np.float64)
    fgd_model1 = np.zeros((1, 65), np.float64)
    bgd_model2 = np.zeros((1, 65), np.float64)
    fgd_model2 = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_small, mask, bg_rect, bgd_model1, fgd_model1, 5, cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(img_small, mask, bg_rect, bgd_model2, fgd_model2, 2, cv2.GC_INIT_WITH_MASK)

    # mask to remove background
    mask_result = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    # if we are removing too much, assume there is no background
    unique, counts = np.unique(mask_result, return_counts=True)
    mask_dict = dict(zip(unique, counts))
    if mask_dict[0] > mask_dict[255] * 1.2:
        mask_result = np.where((mask == 0) + (mask != 1) + (mask != 3), 255, 0).astype('uint8')

    # apply mask to image
    masked = cv2.bitwise_and(img_small, img_small, mask=mask_result)
    masked[mask_result < 2] = [0, 0, 255]  # change black bg to blue

    # save result
    masked = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path + "/" + image_name, masked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Удаление фона с изображений.')
    parser.add_argument('-p', '--path', help='Путь к папке с изображениями',
                        default="img_test", type=str)
    parser.add_argument('-o', '--out', help='Папка для результатов',
                        default="out_test", type=str)
    parser.add_argument('-r', '--bgrectsize',
                        help='Отступу от краев изображения точно считающийся фоном (должен быть > 0.0 и < 0.3)',
                        default=0.02, type=float)
    parser.add_argument('-f', '--fgrect',
                        help='Процент изображения (область в центре) точно считающийся НЕ фоном ( > 0.0 и меньше 0.6)',
                        default=0.4, type=int)
    parser.add_argument('-s', '--resize', help='Изменить размер изображения к заданному (по меньшей стороне)',
                        default=300, type=int)

    args = parser.parse_args()
    inPath = args.path
    outPath = args.out
    bg_rect = args.bgrectsize
    fg_rect = args.fgrect
    resize = args.resize

    for filename in os.listdir(inPath):
        remove_bg(filename, inPath, outPath, bg_rect, fg_rect, resize)
