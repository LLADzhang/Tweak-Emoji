import os
import argparse
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Raw data folder')
    parser.add_argument('--output', type=str, help='Organized data folder')
    args = parser.parse_args()

    classes = ['neutral', 'anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    inDir = args.input
    labelsDir = os.path.join(inDir, 'labels')
    imgsDir = os.path.join(inDir, 'images')

    outDir = args.output
    trainDir = os.path.join(outDir, 'train')
    testDir = os.path.join(outDir, 'test')

    for section in os.listdir(labelsDir):
        if section[0] != '.':
            for seq in os.listdir(os.path.join(labelsDir, section)):
                if seq[0] != '.':
                    for labelFile in os.listdir(os.path.join(labelsDir, section, seq)):
                        if labelFile[0] != '.':
                            labelPath = os.path.join(labelsDir, section, seq, labelFile)
                            fd = open(labelPath, 'r')
                            line = fd.readline()
                            # print(line)
                            label = line.split('.')[0].strip()
                            if classes[int(label)] not in os.listdir(trainDir):
                                os.mkdir(os.path.join(trainDir, classes[int(label)]))
                            if classes[int(label)] not in os.listdir(testDir):
                                os.mkdir(os.path.join(testDir, classes[int(label)]))

                            imgName = labelFile[0:-12] + '.png'
                            imgPath = os.path.join(imgsDir, section, seq, imgName)
                            savePath = os.path.join(trainDir, classes[int(label)], imgName)
                            shutil.copyfile(imgPath, savePath)

    for label in os.listdir(trainDir):
        n = len(os.listdir(os.path.join(trainDir, label)))
        nTest = int(n * 0.2)
        
        for image in os.listdir(os.path.join(trainDir, label))[: nTest + 1]:
            shutil.move(os.path.join(trainDir, label, image), os.path.join(testDir, label, image))

