import sys
import os


def main(data_path, key_list):
    data_path = data_path
    train_path = data_path + '/train'
    valid_path = data_path + '/valid'
    train_list = os.listdir(train_path)
    key_list = key_list
    label_dic = {key_list[0]: '0',
                 key_list[1]: '1',
                 key_list[2]: '2',
                 key_list[3]: '3',
                 key_list[4]: '4',
                 key_list[5]: '5'}

    train_labels = []
    valid_labels = []
    print(label_dic.keys())
    for i in train_list:
        if not i.endswith('.txt'):
            label = label_dic[i]
            train_image_path = train_path + '/' + i
            valid_image_path = valid_path + '/' + i
            train_images_name = os.listdir(train_image_path)
            valid_images_name = os.listdir(valid_image_path)
            for image in train_images_name:
                label_item = i+'/'+image + ' ' + label
                train_labels.append(label_item)
            for image in valid_images_name:
                label_item = i+'/'+image + ' ' + label
                valid_labels.append(label_item)

    print('There are', len(train_labels), 'image in Train Data.')
    print('There are', len(valid_labels), 'image in Validation Data.')

    label_file = train_path + '/labels.txt'
    with open(label_file, 'w') as lf:
        for i in train_labels:
            data = i + '\n'
            lf.writelines(data)
        lf.close()

    label_file = valid_path + '/labels.txt'
    with open(label_file, 'w') as lf:
        for i in valid_labels:
            data = i + '\n'
            lf.writelines(data)
        lf.close()


if __name__ == '__main__':
    data_path = sys.argv[1]
    key_list = sys.argv[2].split(',')
    main(data_path=data_path, key_list=key_list)