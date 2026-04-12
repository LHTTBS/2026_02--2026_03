import json
import csv
from pathlib import Path

# 数据根目录（根据你的实际情况调整）
DATA_ROOT = Path('code\\fakenewsnet_dataset') / 'fakenewsnet_dataset'

# 输出的 CSV 文件路径
OUTPUT_CSV = 'cleaned_news.csv'

# 用于记录缺失文件或空文本的新闻（可选，后续删除用）
missing_folders = []

# 准备写入 CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['source', 'label', 'news_id', 'text', 'title', 'publish_date']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    total = 0
    valid = 0
    missing = 0
    empty_text = 0

    for source in ['gossipcop', 'politifact']:
        for label in ['fake', 'real']:
            label_dir = DATA_ROOT / source / label
            if not label_dir.exists():
                print(f"目录不存在：{label_dir}")
                continue

            for news_dir in label_dir.iterdir():
                if not news_dir.is_dir():
                    continue

                total += 1
                json_file = news_dir / 'news content.json'

                if not json_file.exists():
                    print(f"文件缺失：{json_file}")
                    missing += 1
                    missing_folders.append(news_dir)
                    continue

                with open(json_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"JSON 解析失败：{json_file}")
                        missing += 1
                        missing_folders.append(news_dir)
                        continue

                text = data.get('text', '').strip()
                if not text:
                    print(f"警告：{news_dir.name} 的文本为空")
                    empty_text += 1
                    missing_folders.append(news_dir)  # 也视为无效，可删除重下
                    continue

                # 有效样本
                valid += 1
                writer.writerow({
                    'source': source,
                    'label': label,
                    'news_id': news_dir.name,
                    'text': text,
                    'title': data.get('title', ''),
                    'publish_date': data.get('publish date', '')
                })

print(f"\n统计结果：")
print(f"总新闻文件夹数：{total}")
print(f"有效样本数（有正文）：{valid}")
print(f"缺失 news content.json 数：{missing}")
print(f"文本为空数：{empty_text}")
print(f"\n有效样本已保存至：{OUTPUT_CSV}")