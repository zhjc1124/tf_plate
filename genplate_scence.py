from PlateCommon import *
import os
TEMPLATE_IMAGE = "./images/template.bmp"
# 这里假设车牌最小的检测尺寸是65*21，检测车牌的最小图像为65*21，车牌宽高比变化范围是(1.5, 4.0)
PLATE_SIZE_MIN = (65, 21)


class GenPlateScene:
    '''车牌数据生成器，车牌放在自然场景中，位置信息存储在同名的txt文件中
    '''

    def __init__(self, fontCh, fontEng):
        self.fontC = ImageFont.truetype(fontCh, 43, 0)  # 省简称使用字体
        self.fontE = ImageFont.truetype(fontEng, 60, 0)  # 字母数字字体
        self.img = np.array(Image.new("RGB", (226, 70), (255, 255, 255)))
        self.bg = cv2.resize(cv2.imread(TEMPLATE_IMAGE), (226, 70))
        self.noplates_path = os.listdir('./data/scene')

    def gen_plate_string(self):
        '''生成车牌号码字符串'''
        plate_str = ""
        for cpos in range(7):
            if cpos == 0:
                plate_str += chars[r(31)]
            elif cpos == 1:
                plate_str += chars[41 + r(24)]
            else:
                plate_str += chars[31 + r(34)]
        return plate_str

    def draw(self, val):
        offset = 2
        self.img[0:70, offset + 8:offset + 8 + 23] = GenCh(self.fontC, val[0])
        self.img[0:70, offset + 8 + 23 + 6:offset + 8 + 23 + 6 + 23] = GenCh1(self.fontE, val[1])
        for i in range(5):
            base = offset + 8 + 23 + 6 + 23 + 17 + i * 23 + i * 6
            self.img[0:70, base:base + 23] = GenCh1(self.fontE, val[i + 2])
        return self.img

    def generate(self, text):
        print(text, len(text))
        fg = self.draw(text)  # 得到白底黑字
        # cv2.imwrite('01.jpg', fg)
        fg = cv2.bitwise_not(fg)  # 得到黑底白字
        # cv2.imwrite('02.jpg', fg)
        com = cv2.bitwise_or(fg, self.bg)  # 字放到（蓝色）车牌背景中
        # cv2.imwrite('03.jpg', com)
        com = rot(com, r(60) - 30, com.shape, 30)  # 矩形-->平行四边形
        # cv2.imwrite('04.jpg', com)
        com = rotRandrom(com, 10, (com.shape[1], com.shape[0]))  # 旋转
        # cv2.imwrite('05.jpg', com)
        com = tfactor(com)  # 调灰度
        # cv2.imwrite('06.jpg', com)

        com, loc = random_scene(com, self.noplates_path)  # 放入背景中
        if com is None or loc is None:
            return None, None
        # cv2.imwrite('07.jpg', com)
        com = AddGauss(com, 1 + r(4))  # 加高斯平滑
        # cv2.imwrite('08.jpg', com)
        com = addNoise(com)  # 加噪声
        # cv2.imwrite('09.jpg', com)
        return com, loc

    def gen_batch(self, batchSize, outputPath, xmlPath):
        '''批量生成图片'''
        i = 1
        if not os.path.isdir(outputPath):
            os.makedirs(outputPath)
        if not os.path.isdir(xmlPath):
            os.makedirs(xmlPath)
        if os.listdir(outputPath):
            i = max(map(lambda x: int(x.split('.')[0]), os.listdir(outputPath))) + 1
        while True:
            if i == batchSize + 1:
                return
            plate_str = self.gen_plate_string()
            img, loc = self.generate(plate_str)
            if img is None:
                continue
            cv2.imwrite(outputPath + "/" + str(i).zfill(6) + ".jpg", img)
            print(loc, plate_str, str(i).zfill(6))
            self.gen_xml(i, loc, xmlPath)
            i += 1

    def gen_xml(self, i, loc, xmlPath):
        with open(xmlPath + "/" + str(i).zfill(6) + ".xml", 'w') as obj:
            obj.write('<annotation>\n')
            obj.write('    <folder>VOC2012</folder>\n')
            obj.write('    <filename>%s.jpg</filename>\n' % str(i).zfill(6))
            obj.write('    <source>\n')
            obj.write('        <database>0</database>\n')
            obj.write('        <annotation>1</annotation>\n')
            obj.write('        <image>2</image>\n')
            obj.write('    </source>\n')
            obj.write('    <size>\n')
            obj.write('        <width>1920</width>\n')
            obj.write('        <height>1080</height>\n')
            obj.write('        <depth>3</depth>\n')
            obj.write('    </size>\n')
            obj.write('    <segmented>1</segmented>\n')
            obj.write('    <object>\n')
            obj.write('        <name>plate</name>\n')
            obj.write('        <pose>unspecified</pose>\n')
            obj.write('        <truncated>0</truncated>\n')
            obj.write('        <difficult>0</difficult>\n')
            obj.write('        <bndbox>\n')
            obj.write('            <xmin>%s</xmin>\n' % loc[0])
            obj.write('            <ymin>%s</ymin>\n' % loc[1])
            obj.write('            <xmax>%s</xmax>\n' % loc[2])
            obj.write('            <ymax>%s</ymax>\n' % loc[3])
            obj.write('        </bndbox>\n')
            obj.write('    </object>\n')
            obj.write('</annotation>\n')

def main():
    G = GenPlateScene("./font/platech.ttf", './font/platechar.ttf')
    G.gen_batch(100, './VOCdevkit/VOC2012/JPEGImages', './VOCdevkit/VOC2012/Annotations')


if __name__ == '__main__':
    main()
