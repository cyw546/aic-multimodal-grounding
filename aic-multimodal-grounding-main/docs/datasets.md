# RefCOCO / RefCOCO+ / RefCOCOg 公开数据集说明

## 1. 数据来源
- 图像来源：COCO 2014 train‑val images
- 指代表达标注：RefCOCO family (RefCOCO, RefCOCO+, RefCOCOg)
- 官方项目：https://github.com/lichengunc/refer

## 2. 许可证与使用限制
- COCO图像：CC‑BY‑4.0
- RefCOCO系列标注：MIT License
- 使用约束：本项目**禁止使用比赛初赛官方数据用于训练、人工标注**，仅使用外部公开RefCOCO系列数据集。

## 3. 数据集规模统计

### RefCOCO
- Images：19994
- Total Query：142209
- Train Query：120624
- Val Query：11407

### RefCOCO+
- Images：19994
- Total Query：141564
- Train Query：120191
- Val Query：11385

### RefCOCOg
- Images：26711
- Total Query：104560
- Train Query：85474
- Val‑g：4822；Val‑u：4946

## 4. 原始数据格式说明
1. 图片：COCO2014 jpg，RGB三通道
2. 原始bbox：像素坐标 `[x,y,w,h]`（COCO格式）
3. query：英文自然语言指代表达
4. 原始split划分：train / val / testA / testB

## 5. 处理流程
1. 读取refer库提供的refcoco标注，读取coco图片
2. 将原始 `[x,y,w,h]`像素框转换为 `xyxy`像素：`x2=x+w, y2=y+h`
3. 除以图像宽高，归一化为0‑1之间 `xyxy_norm`
4. 过滤非法bbox（x1>=x2、y1<y2、越界）
5. 输出统一JSONL格式，一行一条样本
6. 生成 train.jsonl、val.jsonl
7. 红外、深度路径置为null（RefCOCO只有RGB，无红外深度）

## 6.输出字段说明
- sample_id：样本唯一编号
- source：数据集来源refcoco
- split：train / val
- query_id：原始标注id
- query：英文指代表达
- visible_path：图片相对路径
- infrared_path: null
- depth_path: null
- bbox: [x1,y1,x2,y2] 归一化0‑1
- bbox_format: "xyxy_norm"
- width、height：原图像素宽高

## 7.注意
>图片、生成的*.jsonl文件不提交Git，仅提交代码脚本、配置、文档、单元测试。