"""界面展示用中文名称。英文类别及其编号仍由 checkpoint 决定。"""
# 键必须与 Oxford-IIIT Pet 官方英文类别名（转为小写后）一致。
# 该字典只影响界面文字，不参与模型训练，也不会改变类别编号。
BREED_ZH = {
    "abyssinian": "阿比西尼亚猫", "american_bulldog": "美国斗牛犬",
    "american_pit_bull_terrier": "美国比特斗牛梗", "basset_hound": "巴吉度猎犬",
    "beagle": "比格犬", "bengal": "孟加拉猫", "birman": "伯曼猫",
    "bombay": "孟买猫", "boxer": "拳师犬", "british_shorthair": "英国短毛猫",
    "chihuahua": "吉娃娃", "egyptian_mau": "埃及猫",
    "english_cocker_spaniel": "英国可卡犬", "english_setter": "英国雪达犬",
    "german_shorthaired": "德国短毛指示犬", "great_pyrenees": "大白熊犬",
    "havanese": "哈瓦那犬", "japanese_chin": "日本狆", "keeshond": "荷兰毛狮犬",
    "leonberger": "兰伯格犬", "maine_coon": "缅因猫", "miniature_pinscher": "迷你杜宾犬",
    "newfoundland": "纽芬兰犬", "persian": "波斯猫", "pomeranian": "博美犬",
    "pug": "巴哥犬", "ragdoll": "布偶猫", "russian_blue": "俄罗斯蓝猫",
    "saint_bernard": "圣伯纳犬", "samoyed": "萨摩耶犬", "scottish_terrier": "苏格兰梗",
    "shiba_inu": "柴犬", "siamese": "暹罗猫", "sphynx": "斯芬克斯猫",
    "staffordshire_bull_terrier": "斯塔福郡斗牛梗", "wheaten_terrier": "软毛麦色梗",
    "yorkshire_terrier": "约克夏梗",
}


def breed_zh(name):
    """返回品种中文名；未知名称则转换为便于阅读的英文标题格式。"""
    return BREED_ZH.get(name.lower(), name.replace("_", " ").title())
