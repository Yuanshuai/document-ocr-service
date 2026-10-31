import re


class ParseOCRDirector(object):

    # 忽略页眉页脚
    @staticmethod
    def ignore_header_footer(paragraph):
        reg3 = "^[第]\d+[\s\n]*.{1,10}页$|^[IVXLCDMivxlcdmⅠⅡⅢⅩⅤⅬⅭⅮⅯⅣⅥⅦⅧⅨⅫ]+$|^\d{1,3}.{1,10}页$"

        return re.search(reg3, paragraph)

    @staticmethod
    def filter_paragraph(paragraph):
        reg = "^[一二三四五六七八九十第(（附、].{2,200}|^\d+[^\d]+.{2,100}"
        # 非符合开头为字符的和结尾为数字的过滤掉
        reg2 = "^[^一二三四五六七八九十第(（附、].*\d$"

        # 过滤页码
        reg3 = "^[第]\d+[\s\n]*.{1,10}页$|^[IVXLCDMivxlcdmⅠⅡⅢⅩⅤⅬⅭⅮⅯⅣⅥⅦⅧⅨⅫ]+$|^\d{1,3}.{1,10}页$"
        is_page = re.search(reg3, paragraph)
        if is_page:
            return False

        # 2.开头字符[一二三四五六七八九十第(（附]
        is_had = re.search(reg, paragraph)
        if is_had:
            return True

        # 1.存在Error的过滤掉
        is_had_error = re.search(r"error.{0,10}Bookmark", paragraph, re.I)
        if is_had_error:
            return False

        is_had_num_end = re.search(reg2, paragraph, re.I)
        if is_had_num_end:
            return False

        return False

    # 判断前后是否一致
    @staticmethod
    def verify_paragraph(pre, next):
        patterns = ["^[一二三四五六七八九十].*", "^[第(（附、].*", "^\d+[^\d]+.*"]

        if re.search(r'^附.*', next):
            return True

        for pattern in patterns:
            is_pre = bool(re.search(pattern, pre))
            is_next = bool(re.search(pattern, next))

            if is_pre and is_next:
                return True

        return False

    def pre_and_next_is_valid(self, pre, next) -> bool:
        # 如果是附件的话直接返回True
        if re.search(r'^附.*', next) or re.match(r'^目[\s\n]*录[\s\n]*$', pre):
            return True
        # 1.判断两个格式是否一致
        # is_some = self.are_formats_same(pre, next)
        # if not is_some:
        #     return False
        # 获取值
        pattern = r'^第([一二三四五六七八九十]+).+'
        pattern2 = r'^[（(]?([一二三四五六七八九十]+)[、)）]'
        number_pattern = r'^([\.]?\d*[\.]?\d*)+'
        match1 = re.match(pattern, pre)
        match2 = re.match(pattern2, pre)
        match3 = re.match(number_pattern, pre)

        try:
            if match1:
                next_match = re.match(pattern, next)
                if not next_match:
                    next_match = re.match(pattern2, next)
                if not next_match:
                    next_match = re.match(number_pattern, next)

                next_num = self.chinese_to_arabic(next_match.group(1))

                if not next_num:
                    next_num = re.sub(r'^[\.\s]+|[\.\s]+$',
                                      '', next_match.group())
                pre_num = self.chinese_to_arabic(match1.group(1))
                # next_num = self.chinese_to_arabic(next_match.group(1))

                return str(next_num) > str(pre_num)

            elif match2:
                next_match = re.match(pattern2, next)
                pre_num = self.chinese_to_arabic(match2.group(1))
                next_num = self.chinese_to_arabic(next_match.group(1))

                return str(next_num) > str(pre_num)
            elif match3:
                pre_num_str = re.sub(r'^[\.\s]+|[\.\s]+$', '', match3.group())
                next_num_str = re.sub(
                    r'^[\.\s]+|[\.\s]+$', '', re.match(number_pattern, next).group())
                # pre_num = float(pre_num_str)
                # next_num = float(next_num_str)
                return next_num_str > pre_num_str
            else:
                return False
        except Exception as e:
            print(pre)
            print(next)
            print(e)
            return False

    @staticmethod
    def chinese_to_arabic(chinese_num: str) -> int:
        chinese_numerals = {'零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                            '百': 100, '千': 1000, '万': 10000, '亿': 100000000}

        total = 0
        r = 1  # 表示单位：个位、十位、百位等
        for i in range(len(chinese_num) - 1, -1, -1):
            val = chinese_numerals.get(chinese_num[i])
            if val >= 10 and i == 0:  # 处理"十五"这样的情况
                if r > 1:
                    total += 10 * r
                else:
                    total += 10
            elif val >= 10:
                r = val
            else:
                total += r * val
        return total

    @staticmethod
    def get_chapter_format(chapter):
        if chapter.startswith("第"):
            return "Chinese"
        elif re.search(r'^[（()]', chapter):
            return "Parentheses"
        elif re.search(r'^\d', chapter):
            return 'Number'
        elif re.search(r'^[一二三四五六七八九十]+', chapter):
            return "Capital"
        elif chapter.isdigit():
            return "Arabic"
        else:
            return "Other"

    def are_formats_same(self, chapter1, chapter2):
        format1 = self.get_chapter_format(chapter1)
        format2 = self.get_chapter_format(chapter2)
        return format1 == format2

    @staticmethod
    def update_end_paragraph(backup_end_paragraph, start_paragraph):
        return {
            **backup_end_paragraph,
            'y': backup_end_paragraph["y"] + start_paragraph["y"],
            'page': start_paragraph['page']
        }

    def read_and_process_json(self, data):
        data = self.data_correction(data)
        # 收集每一页的收尾坐标数据
        collections = []

        start_paragraph = {
            "x": 0,
            "y": 0,
            "page": 0,
            "paragraph": ""
        }
        end_paragraph = {
            "x": 0,
            "y": 0,
            "page": 0,
            "paragraph": ""
        }
        # 结束坐标，以防 end 为空过滤不了
        backup_end_paragraph = {
            **end_paragraph,
            "y": 100
        }

        y = 0  # y轴
        paragraph = ""  # 当前行内容
        min_x = 100000
        max_x = 0

        # 收集目录
        directors = []
        # 如果大于3并且是同一页那么归为目录
        is_directors = []

        # 计停止数
        stop_index = 0
        pre_item = {
            "x": 0,
            "y": 0,
            "page": 0,
            "paragraph": ""
        }

        # 加个y轴偏移量
        offset_y = 15

        for item in data:

            # 只有确定到目录以后才开始计算最小的x
            if start_paragraph["x"] > 0:
                min_x = min(min_x, item["x"])
                max_x = max(max_x, item["x"] + item['width'])

            # 矫正偏移量
            if len(paragraph) > 0 and y > 0 and y + offset_y > item["y"] and item['y'] > y - offset_y:
                item['y'] = y

            if item["y"] != y or item['page'] != pre_item['page']:
                y = item["y"]

                paragraph = paragraph.strip()

                # if self.ignore_header_footer(paragraph):
                #     paragraph = ""
                #     continue

                # print("paragraph-----", [paragraph, pre_item["y"], pre_item['page']])
                # 判断是否在这段收集目录
                is_collection_directory = False
                # 判断这一行是否是目录
                if start_paragraph["x"] == 0:
                    search_data = re.search(r"^目[\s\n]*录[\s\n]*$", paragraph)
                    if search_data:
                        min_x = item['x']
                        start_paragraph["x"] = item['x']
                        start_paragraph["y"] = pre_item["y"]
                        start_paragraph["page"] = pre_item["page"]
                        start_paragraph["paragraph"] = paragraph
                        directors.append(start_paragraph)
                        is_collection_directory = True

                # 识别到目录才开始收集
                if not is_collection_directory and start_paragraph["x"] > 0 and len(paragraph) >= 2:
                    # 过滤段落是否符合要求
                    is_directory = self.filter_paragraph(paragraph)

                    # 1.如果是跟目录是同一页那么全过滤掉只选最后一个
                    if start_paragraph["page"] == pre_item["page"]:
                        if is_directory:
                            director_item = {
                                "x": max_x,
                                "y": pre_item["y"] + pre_item['height'],
                                "page": pre_item["page"],
                                "paragraph": paragraph
                            }
                            directors.append(director_item)
                            end_paragraph = director_item

                            start_paragraph['x'] = min_x

                    else:
                        # 不是同一页那么判断是否符合,这里是处理第二页的第一行
                        if is_directory and len(is_directors) == 0 and self.pre_and_next_is_valid(
                                directors[-1]['paragraph'], paragraph):
                            is_directors = []
                            # 收集上一页的数据
                            collections.append({
                                "start_paragraph": start_paragraph,
                                "end_paragraph": end_paragraph['y'] > 0 and end_paragraph or self.update_end_paragraph(backup_end_paragraph, start_paragraph)
                            })

                            start_paragraph = {
                                "x": item['x'],
                                "y": pre_item["y"] + pre_item['height'],
                                "page": 0,
                                "paragraph": paragraph
                            }

                            min_x = 10000
                            max_x = 0

                            # start_paragraph['x'] = min_x
                            is_directors.append(start_paragraph)

                            stop_index = 0

                        elif len(is_directors) > 0 and stop_index < 2:
                            if is_directors and self.pre_and_next_is_valid(is_directors[-1]['paragraph'], paragraph):
                                directory_item = {
                                    "x": min_x,
                                    "max_x": max_x,
                                    "y": pre_item["y"] + pre_item['height'],
                                    "page": pre_item["page"],
                                    "paragraph": paragraph
                                }
                                is_directors.append(directory_item)

                                if len(is_directors) >= 2:
                                    start_paragraph['page'] = pre_item["page"]
                                    directors += is_directors
                                    stop_index = 0
                                    is_directors = []

                            else:
                                stop_index += 1
                        elif len(is_directors) == 0 or stop_index >= 2:
                            if len(is_directors) == 0:
                                start_paragraph['x'] = min_x
                                # 不符合直接收集结束
                                collections.append({
                                    "start_paragraph": start_paragraph,
                                    "end_paragraph": end_paragraph['y'] > 0 and end_paragraph or self.update_end_paragraph(backup_end_paragraph, start_paragraph)
                                })

                            break

                pre_item['height'] = item['height']
                pre_item["y"] = y
                pre_item["paragraph"] = paragraph
                pre_item["page"] = item["page"]
                paragraph = ""

            paragraph += item["char"]

            # 处理特殊情况，如果最后一页是目录
            if data[-1] == item and start_paragraph['x'] > 0 and end_paragraph['x'] > 0:
                if self.pre_and_next_is_valid(directors[-1]['paragraph'], paragraph):
                    end_paragraph = {
                        "x": max_x,
                        "y": item["y"] + item['height'],
                        "page": item["page"],
                        "paragraph": paragraph
                    }

                collections.append({
                    "start_paragraph": start_paragraph,
                    "end_paragraph": end_paragraph['y'] > 0 and end_paragraph or self.update_end_paragraph(backup_end_paragraph, start_paragraph)
                })

        # print("start_paragraph----", [start_paragraph] )
        # print("end_paragraph----", [end_paragraph] )
        # print("directors----", directors )
        # print("collections----", collections )
        return {
            # "start_paragraph": start_paragraph,
            # "end_paragraph": end_paragraph,
            "collections": collections,
            # "directors": directors
        }

    # 数据矫正
    @staticmethod
    def data_correction(page_result):
        new_result = []
        pre_item = {'x': 0, 'y': 0, 'page': 1}  # 为了避免在第一次循环时 pre_item 未定义
        offset_y = 15
        for item in page_result:
            # 错位矫正
            if pre_item['page'] == item['page'] and pre_item['x'] > 0 and pre_item['y'] + offset_y > item["y"] and item['y'] > pre_item['y'] - offset_y and \
                    pre_item['x'] > item['x']:
                new_result[-1] = item
                item = pre_item
            pre_item = item
            new_result.append(item)
        return new_result
