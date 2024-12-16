import csv
import os
import json
import re
import string
from pprint import pprint


def get_data_by_field(book_data, field):
    """根据字段名称获取书籍数据中指定字段的所有内容"""
    return [e[field] for e in book_data['fields'] if e.get(field) is not None]

def extract_title(book_data):
    """提取书籍标题"""
    values = get_data_by_field(book_data, '245')
    assert len(values) == 1
    title = []
    c_end_char = None
    has = False
    for e in values[0]['subfields']:
        if 'c' in e:
            c_end_char = e['c'][-1]
            if c_end_char not in string.punctuation:
                c_end_char = None
            if '=' in e['c'] and '/' in e['c']:
                idx1 = e['c'].index('=')
                idx2 = e['c'].index('/')
                if -1 < idx1 < idx2:
                    if len(title) > 0 and title[-1].endswith('/'):
                        title[-1] = title[-1][0:-1].strip()
                    title.append("= " + (e['c'][idx1 + 1:idx2].strip()))
                    has = True
            continue
        sub_values = [se for se in e.values()]
        assert len(sub_values) == 1
        if c_end_char:
            title.append(c_end_char)
            c_end_char = None
            has = True
        title.append(sub_values[0].strip())
    title = ' '.join(title)
    title = title.strip()
    if title.endswith('/'):
        title = title[0:-1].strip()
    return title


def extract_abstract(data):
    """提取摘要"""
    values = get_data_by_field(data, '520')
    abstract = []
    for value in values:
        ind1 = value['ind1'].strip()
        if not (ind1 == '' or ind1 == '3'):
            continue
        for e in value['subfields']:
            sub_values = [se for se in e.values()]
            assert len(sub_values) == 1
            abstract.append(sub_values[0].strip())
    return '\n'.join(abstract)



def lcc_standarize(lcc):
    # idx, end_idx = -1, -1
    # for i in range(0, len(lcc)):
    #     if lcc[i].isalpha():
    #         idx = i
    #     else:
    #         break
    # if idx>-1:
    #     for i in range(idx, len(lcc)):
    #         if lcc[i].isdigit() or lcc[i]=='.':
    #             idx = i
    #         else:
    #             if lcc[i-1] == '.' and lcc[i].isalpha():
    #                 end_idx = i-1
    #                 break
    # if end_idx > -1:
    #     lcc_std = lcc[0:end_idx]
    # else:
    #     lcc_std = lcc
    lcc_std = re.match(r'^([A-Za-z]+)\W*(\d+(\.\d+)?)', lcc)
    if lcc_std:
        lcc_std = lcc_std.group()
        lcc_std = lcc_std.upper().replace(" ", "")
    else:
        lcc_std = ''
        # print(lcc)
    return lcc_std


def extract_lcc(data):
    """提取国会图书馆分类号"""
    values1 = get_data_by_field(data, '050')
    values2 = get_data_by_field(data, '090')
    values = values1+values2
    lccs = []
    lccs_std = []
    for value in values:
        for e in value['subfields']:
            if 'a' in e:
                sub_values = [se for se in e.values()]
                assert len(sub_values) == 1
                lcc = sub_values[0].strip()
                lccs.append(lcc)
                lccs_std.append(lcc_standarize(lcc))
    return ' ; '.join(lccs), ' ; '.join(lccs_std)


def extract_table_of_contents(data):
    """提取目录"""
    values = get_data_by_field(data, '505')
    tocs = []
    for value in values:
        for e in value['subfields']:
            sub_values = [se for se in e.values()]
            assert len(sub_values) == 1
            tocs.append(sub_values[0].strip())
    assert len(tocs) > 0
    return '\n'.join(tocs).strip()


def extract_publisher_year(data):
    """提取年份"""
    values = get_data_by_field(data, '008')
    if len(values) > 0:
        # if values[0][7:11] !=  values[0][11:15] and values[0][11:15].strip():
        #     print(values)
        return values[0][7:11].strip(), values[0][11:15].strip()
    else:
        values = get_data_by_field(data, '264')
        assert len(values) == 1
        if values[0]['ind2'] == '1':
            for e in values[0]['subfields']:
                if 'c' in e:
                    sub_values = [se for se in e.values()]
                    assert len(sub_values) == 1
                    year = re.search(r'\d+', sub_values[0]).group()
                    # print("------")
                    # print(sub_values[0], year)
                    return year, ''
        return None, None


def extract_subject_headings(book_data):
    """提取主题词"""
    fields = [
        '650',  # Topical subject heading
        #'600',  # Personal name
        #'610',  # organizational name
        #'611',  # meeting name
        #'630',  # Title
        #'655',  # Genre heading
        #'647',  # Named event
        #'648',  # chronological term
        #'651',  # geographic name
    ]
    subjects_lcsh = []
    subjects_fast = []

    for field in fields:
        values = get_data_by_field(book_data, field)
        for value in values:
            # 检查是否为 LCSH 或 FAST 主题词
            if value['ind2'].strip() == '0':  # LCSH headings
                subject = [e['a'].strip() for e in value['subfields'] if 'a' in e]
                subjects_lcsh.extend(subject)
            elif value['ind2'].strip() == '7':  # 可能是 FAST headings
                # 检查 subfields 中是否有 2 字段，并且值为 'fast'
                fast_indicator = [e['2'].strip().lower() for e in value['subfields'] if '2' in e]
                if fast_indicator and 'fast' in fast_indicator:
                    # 提取 'a' 字段作为 FAST 主题词
                    subject = [e['a'].strip() for e in value['subfields'] if 'a' in e]
                    subjects_fast.extend(subject)
                    print(f"提取到的 FAST 主题词: {subject}")  # 打印提取到的 FAST 主题词
                

    return '; '.join(subjects_lcsh), '; '.join(subjects_fast)


def extract_bibli(data_dir, save_file_path):
    with open(save_file_path, mode='w', newline='', encoding='utf-8-sig') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['lcc', 'lcc_std', 'start_year', 'end_year', 'title', 'abstract', 'toc', 'lcsh_subject_headings', 'fast_subject_headings'])
        
        filenames = os.listdir(data_dir)
        filenames.sort()
        print(f"总共 {len(filenames)} 个文件待处理")  # 输出待处理文件总数

        count = 0  # 计数器，用于统计成功写入的行数
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(data_dir, filename)
            print(f"正在处理文件: {filename}")  # 打印当前处理的文件名

            try:
                with open(file_path, mode='r', encoding='utf-8') as infile:
                    data = json.load(infile)
                    
                    # 检查是否是一个列表，如果是，表示有多本书籍数据
                    if isinstance(data, list):
                        for book_data in data:
                            process_and_write_book_data(writer, book_data)
                            count += 1
                    else:
                        # 如果不是列表，处理单本书籍
                        process_and_write_book_data(writer, data)
                        count += 1
                        
                    print(f"成功写入第 {count} 行")  # 输出当前成功写入的行数
            except Exception as e:
                print(f"处理文件 {filename} 时出现错误: {e}")  # 打印错误信息

    print(f"提取完成，共成功处理 {count} 个文件")
    


def process_and_write_book_data(writer, book_data):
    """处理单个书籍数据并写入 CSV"""
    try:
        title = extract_title(book_data)
        abstract = extract_abstract(book_data)
        lcc, lcc_std = extract_lcc(book_data)
        toc = extract_table_of_contents(book_data)
        syear, eyear = extract_publisher_year(book_data)
        lcsh_subject_headings, fast_subject_headings = extract_subject_headings(book_data)
        
        writer.writerow([lcc, lcc_std, syear, eyear, title, abstract, toc, lcsh_subject_headings, fast_subject_headings])
    except Exception as e:
        print(f"处理书籍数据时出现错误: {e}")
    

if __name__=='__main__':
    data_dir = '/mnt/llm4cat/data-20241004/eng-summary-and-toc'
    save_file_path = '/home/jl1609@students.ad.unt.edu/bibli2.csv'
    extract_bibli(data_dir, save_file_path)
    print("ok")
    # print(lcc_standarize('Z711.6.G46'))