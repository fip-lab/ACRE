#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : tta.py
@Author  : huanggj
@Time    : 2023/5/27 11:13
"""
# a = ['屈己', '容易', '左', '以静', '大多', '臣诚', '别', '先派', '献策', '统军', '最多', '亲眼', '如', '专程', '娴熟', '派士兵', '简约', '相应', '瞬间', '善于', '郁郁而终', '三元', '凶猛', '一个个', '骡子', '亲率', '违规', '故文', '不论是', '树下', '因功', '正气', '莫不', '楚', '顾时率', '轻轻', '自始至终', '一方面', '痛殴', '裕王', '一家', '一块', '常自', '将要', '魏征', '册', '为官者', '并非', '武', '终能', '无从', '正要', '慎重', '按理', '浴乎', '快要', '节操', '尽力', '清高', '真正', '薛平', '致阳', '节夫', '不停', '仓促', '主帅', '白骨', '不得行', '以年', '相机', '狐狸', '隐居', '凡是', '后泛', '径自', '裕任', '和睦', '杨沛', '威震', '习凿', '应先', '恰恰', '官至', '洪州', '为士', '总共', '何谈', '只好', '预', '有谦', '谨', '附势', '对比', '一贯', '相士', '逐步', '扮成', '对贩', '郴州', '服药', '哽咽', '萧宏', '牛弘', '有的放矢', '铖想', '广泛', '盛唐', '到', '深为', '多谦', '撒离', '意外', '英勇', '恐怖', '泣告', '年老', '乘船', '早早', '努力', '派军', '有幸', '当楚', '兼选', '高度', '新帝', '下江', '一地', '标记', '身陷', '处处', '因言', '严令', '听说官', '平生', '诫堂', '守城', '四字', '审慎', '反倒', '亲笔', '千万', '囚妻', '毫不', '贸然', '大怒', '平易', '腹中', '满眼', '全线', '仍心', '事于', '招民', '未曾', '时均', '大规模', '从谏', '偶然', '碌', '险临', '减饷', '贫困', '诏厚', '明帝', '并没有', '贪念', '揭露', '阴险', '陆续', '临死', '赋词', '之前', '认王', '越境', '一片石', '悉心', '因故', '不管是', '全文', '违法', '谏言', '放诞', '上', '然', '那种', '极富', '潇洒', '诚果', '单骑', '连带', '宽厚', '恰', '明确', '亲下', '借梁', '虚乌', '范雍', '尤其', '一同', '同样', '微服', '人杀', '事构', '千方百计', '可能', '不智', '永平', '有贬', '遇事', '祖攻', '邪恶', '以民', '适当', '恰巧', '临财', '忍辱', '统一', '晚景', '匈奴人', '定时', '徒步', '不禁', '姜维', '温和', '超所', '哀怜', '辄', '从小', '依礼制', '奏章', '由衷', '永', '殊死', '镇阳', '要不', '看起来', '降职', '惩恶', '真', '无用', '准时', '事先', '高崧', '贺娄子', '拦道', '亲临', '申时', '时人', '太傅', '伍', '稀自', '被迫', '紧紧', '成遵', '文能', '少', '唯', '刘劭', '随王', '假使', '不务', '无比', '机警', '力图', '勤政', '藉此', '予', '奏皇', '章纶', '一大早', '棒', '都会', '以期', '大体上', '千', '有据地', '唆使', '陈懋', '顾时', '真宗', '尽散', '耕读', '往下', '缓慢', '平稳', '相', '单独', '双目', '行将', '酒间', '陆抗', '全力', '才识', '倒是', '大败古', '广受', '翻新', '似', '勤读', '其要', '深数', '仔细', '蛮夷', '贤', '所以', '萧恢', '不快', '仅仅', '对此', '不料', '豪爽', '辛亥', '抚臣', '因撰', '以利', '诏任', '虔地', '未加', '即便是', '时下', '可惜', '晚', '残酷', '勉力', '相虞', '虞侯', '不敬', '既勇', '君命', '凌策', '筠州', '隧入', '攻', '日久', '悟解', '挺身', '然而', '不拘', '为人', '自纳', '驿使', '正堂', '就座', '不弃', '王镕', '以足', '答解', '实后', '已', '扬干', '顾璘', '早晚', '惮', '襄城', '这是', '奏疏', '突出', '裴侠', '亲密', '冷箭', '太', '元好问', '沐晟', '银朱', '无暇', '从业', '坦率', '伤寒', '拔出', '无法', '即', '任车', '尾', '先览', '广', '重武', '原来', '平准', '特意', '不忍', '近', '惨败', '斌', '谨守', '将帅', '攀权', '传给', '系结', '生前', '在士', '恩准', '奏罢', '与其', '屡屡', '俨然', '正是', '绯衣', '明于', '任后', '顺利', '此举', '即送', '全都', '均赋', '渐渐', '一次', '约', '他用', '安全', '政学', '重点', '特殊', '几', '只身', '一面', '平日', '各自为战', '据为己有', '理性', '明辨', '每当', '垦', '刚强', '侯益', '孝伯言', '陈瑛', '实所', '不愧', '陡然', '可持续', '杨腾', '厉声', '亲自', '向', '亢直', '以西', '缜密', '现', '大帅', '较为', '潜心', '重金', '不大', '永业', '这样', '随风', '有情有义', '肆意', '拟制', '先奏', '恬然', '写文', '责令', '宦官们', '乱兵', '卢秉', '恶意', '待人', '俭朴', '墓时', '辞官', '世让', '忧国', '竟然', '曾师', '他年', '从此', '年', '没', '本该', '不备', '远远', '不止', '终不曾', '通晓', '来得及', '刚好', '尽责', '什么', '闭塞', '几乎', '忽辛', '为何', '着力', '不雅', '因人', '合天', '沽酒', '从今以后', '招讨', '茅屋草舍', '卒章', '优秀', '兵少', '结果', '极力', '仍然', '同', '常安民', '实际上', '琪出', '遣使', '品评', '中上', '四散', '而弟', '非法', '阴碌', '不懈', '蹇义', '仅', '所求', '司录', '慷慨', '镇恶', '乘虚', '塘江', '往昔', '连', '不胜任', '以此', '四处', '固执', '早攻', '专业', '永王', '兴城', '宦历', '紧接着', '说明庄', '徐稚', '勤奋', '可作', '并肩', '刻意', '迟迟', '久', '来不及', '仰', '苦苦', '擅自', '亲善', '终以', '比如', '年富', '先人', '狄青', '寨', '同亨', '还', '清正', '烹杀', '肃敬', '执法', '起起伏伏', '严密', '终当', '有空', '充分', '淳于', '娄谅', '不到', '赵胤', '非任人', '偏', '伪装', '侄', '怎么', '佳', '怒', '安慰', '刻刻', '虽位', '大体', '仍免', '相武', '碍于', '白残', '多施', '如诗如画', '便', '痴', '向上', '柳', '不服', '谦进', '非学', '反遭', '侯霸', '不难', '随机', '一开始', '应予', '成功', '开都', '终日', '杨震', '仇池', '在外', '依次', '有些', '何荣', '永久', '肯定', '亲阅', '趁乱', '博闻强识', '除此之外', '以元', '跟随', '不错', '房琯', '自残', '习;', '相忌', '从来', '嘶鸣', '朝廷', '独', '良弼', '相后', '龚胜', '数罟', '独立', '顺畅', '高梁', '冷落', '不管', '一族', '极尽', '提早', '名垂', '趁百姓', '一生为国', '皆', '一心', '表辞', '粲贬', '杨宽', '迟早', '柳虬', '秘阁', '更为', '许论', '沉溺', '平凡', '严厉', '清楚', '终爱', '轻慢', '私人', '范镇', '各自', '渐善于', '昼夜', '只能', '合计', '入仕', '初步', '受祸', '坦白', '范于', '奴仆', '未尝', '长福', '不易', '深入', '襄乐', '就此', '每天', '无力', '尽', '掾吏', '均匀', '丞', '随时', '敢乱', '孝孺', '就要', '固边', '何晏', '杨雄', '同日', '大量', '阎义', '嫉恨', '当中', '壶头', '马上', '休致', '跟风', '轮流', '赏识', '假托', '小时候', '高俭', '介子', '全身', '难经', '倒', '当宇', '远水', '将校', '对民', '大', '日后', '密请', '当世', '依仗', '经冉', '单于', '实际', '妥帖', '相面', '五堡', '如何', '焚书', '不经', '侬智', '郡公', '除下', '小', '李程', '反之', '以病', '即位', '一会儿', '时', '以资', '极易', '高调', '以爱', '继隆', '实者', '从医', '因而', '不意', '照实', '一如', '依调', '向来', '妙', '此诚', '先天性', '比较', '大举', '白', '沅弟', '受此', '共计', '自请', '终年', '最为', '如称', '许仲', '随张', '分给', '许楫', '超然', '鸿门', '尚文', '不堪', '童仆', '坚强', '轻松', '授职', '神武', '从不拉帮', '常何', '从中', '贬司', '悠闲', '护堤', '卢坦', '相随', '日益', '快', '故入', '常年', '剑南', '离职', '死时', '自然而然', '不惧权贵', '操持', '转而', '那么', '诬陷', '自取', '大幅', '从', '佞臣', '臣犯', '沉着', '勇有', '耻辱', '细致', '迅速', '大兴', '自愿', '私闯', '举身', '安期', '见魏', '可所部', '正', '早些', '中求', '清新', '多', '因古', '虚心', '太过', '元军', '应劭', '曾子', '祖厚', '不辞劳苦', '年出', '借此', '无忌', '有朝一日', '实实在在', '足见', '塘西', '不断', '膂力', '程喜', '敖', '仓廪', '人诬', '立像', '起用', '勇为', '尽职', '此外', '提前', '明显', '好意', '傲慢', '随身', '以谷', '著书', '所到之处', '独守', '荆嗣', '精灵', '何福', '敌将', '尽量', '悲戚喜悦', '语重心长', '无屈', '误', '有点', '立即', '变通', '为国', '到底', '眠', '舒适', '帝进', '方可', '每', '可望', '奸臣', '一挥而就', '怎样', '随事', '日醉', '边', '卧病', '客者', '实则', '每次', '封赐', '困顿', '孤军', '以理', '而且', '孝谐', '自发', '足出', '兵士', '当面', '诵杜', '徐俭', '渐', '顺时', '无可', '一众', '依势', '还常', '差点', '径', '全面', '阐', '不法', '部分', '粤地', '相国', '反复', '分辨', '正侧', '黄宗旦', '只免于', '广构', '毅然决然', '从叔', '类比', '大约', '到处', '矢志', '诬', '陈涉', '索要', '帝安', '相与', '程婴', '相互', '拱手', '篡夺', '联手', '不可', '大胆', '任职于', '一下子', '璐年', '为政', '阴呼', '乞', '即帝', '勤', '专心', '突兀', '逐渐', '佗恃', '究竟', '璨', '高声', '赐御笔', '属下', '悲恸', '古', '更加', '严武', '秘密', '奏折', '俱', '独撑', '祺案', '暂时', '不求', '冒死', '尚', '妾', '故', '尹益', '恭敬如', '慷慨陈词', '随人', '况且', '清明', '大加', '同甘共苦', '永存', '再败', '其随', '非但', '翰于', '不单', '后乐', '一直', '本来', '悟并', '祭神', '勤勉', '无利', '可见', '望山', '极', '不及', '很快', '不媚', '何士', '大盜', '为此', '完好', '消极', '日食', '终老', '愤而', '过量', '缚后', '率子', '不仅', '一致', '深荷', '榻边', '迫赠', '随蔡', '终获', '遭斥', '争相', '正反两面', '大夏', '严词', '一时间', '顺承', '只是', '但', '一时', '沈重', '密谋', '难', '慢慢', '宽宏', '艰苦', '后袭', '急于', '宽宏大度', '热心', '写讽', '好马', '恰好', '独自', '忽然', '勤加', '堂弟', '朱泚', '首先', '此人', '不便', '孝宗', '以致于', '授御', '举止', '帘幕', '办案', '飘逸', '终失', '坛', '一级', '大获', '先行', '独专', '活活', '诏', '照常', '启奏', '诏辞', '分别', '欺辱', '优先', '从西', '全程', '超重', '司马伦', '另一方面', '还祸', '才能', '此后', '史记事', '联车', '万分', '昭亮', '首次', '事后', '从表', '冗滥', '信义;', '赶紧', '月饼', '几近', '严重', '自觉', '强行', '次序', '草率', '轻信', '不必', '无不', '光', '积极', '一时之选', '上委', '钦若', '降卒', '常用', '责骂', '以', '很少', '坦然', '笑出', '薛颜', '年间', '外有', '果断', '依附', '遂', '当刘', '重诲', '挺', '银州', '随性', '往往', '严肃', '义前', '平常', '苑中', '还算', '不能不', '引兵', '公叔', '服徒', '清代', '如镜', '原', '立法令', '凭军', '装殓', '独揽', '招安', '笃定', '丘福', '颇为', '艰险', '斩杀', '初氏', '稍长', '纷纷', '赵王', '另', '熙', '更改', '不等', '一己', '立张', '将近', '襄王', '总', '陆深', '突围', '相平', '宾客', '先心', '蛮', '下次', '始皇', '尽灭', '急剧', '先前', '精诚', '以至于', '诸弟', '尽孝', '应召', '凡事', '白衣', '厉叱', '一瞬间', '敷衍', '许褚', '羊数', '仇理', '惹怒', '孔嵩', '定期', '依稀', '到任', '年少', '双双', '不乐', '奸乱', '矜持', '轻率', '时时', '荐贤', '罪当', '终究', '宦官', '刑罚', '年年', '相反', '适时', '无逸', '多加', '非常', '薛综', '大义', '自此', '因上', '愤懑', '御史安', '随李', '阴氏', '高继升', '伤心', '一并', '稍加', '能干', '为辅', '公堂', '宽松', '疾为', '而', '倾心', '姚崇', '顾雍', '多士', '器重', '乃', '终身', '空怀', '九鼎', '本次', '严刑', '永为官', '晏子', '全义', '帝攻', '神气', '有理', '激烈', '最先', '明亮', '何妥', '那年', '单用', '随', '嵇绍', '置石', '满', '几度', '加上', '佐', '无拔', '赐诗', '一向', '一军', '加官', '正词', '尽数', '隐士', '叛国', '大溃', '坚遣', '正北', '至死不移', '决不', '顾荣', '彻底', '赟', '垦荒', '高低', '外', '扩义', '兵队', '既', '果然', '赃款', '一齐', '事前', '显著', '绮丽', '偶得', '殷武', '谷关', '快马', '士类', '例如', '不复', '大概', '越', '早', '静处', '不再', '先遣', '盗贼', '欧阳', '纪瞻', '对外', '以景', '首', '各家', '密切', '即使', '历阳', '还是', '不但', '亲往', '不纳', '正直', '唯有', '借宴', '相请', '恩泽', '履历', '任', '要不然', '惊险', '叛军', '勤苦', '已往', '因物', '奸豪', '贬职', '通识', '乱政', '诚挚', '经常', '似的', '和好', '恩素', '较', '昂', '因怒', '潞州', '继', '普安', '未遂', '极度', '轻刑', '便宜', '不依', '不住', '虽长', '温峤', '秉性', '几千字书', '长流', '集中', '则', '其用', '宗重', '每日', '自许', '颖悟', '近年来', '一身', '过', '难免', '霸州', '薛莹', '与否', '何卿', '反前', '永徽', '照此', '一整天', '虚', '明白', '各州', '大尹', '远道', '昭雪', '每逢', '何况', '比对', '临危', '任开', '细读', '绝', '轻徭', '不宜', '静辅', '安睡', '北宋', '极为', '元年', '向善', '尊崇', '一天天', '顺从', '自如', '不贤者', '简于', '商鞅', '一忠君', '将士', '翰荐', '一心守节', '诏狱', '过度', '聪明', '屡挫', '以至', '镇州', '余寇', '团团', '殷勤', '一屈于', '以东', '相王', '外夷', '干眼', '紧密', '随众', '务必', '众', '突厥', '舒化', '因饥', '超脱', '因病', '将情', '长', '朱熹', '所', '杖', '其余', '宽待', '早已', '随髙', '顾琛', '每餐', '痛斥', '从犯', '侯景', '执己', '纲常', '恳请', '笔力', '随之', '自主', '路', '斩首', '表面上', '灌孟', '诬蔑', '帅令', '陈蕃', '非议', '篇奏', '初入', '纵任', '有景', '蒙父', '急速', '顾名思义', '先志', '降敌', '一世', '纸', '竭心', '如江', '往今', '目崇', '负', '州府', '素', '故意', '据此', '向下', '不顾一切', '所选', '破', '据险', '巧', '这种', '诏书', '积累', '深沉', '松树', '周莹', '帝灭', '许泰', '惊奇', '统兵', '差点儿', '只', '均施', '世俗', '或黜', '以少', '莫', '才学', '又', '近来', '剿灭', '好心', '勋劳', '庞参', '尚义', '平均', '暂', '早作', '间接', '最好', '总算', '无偿', '绝皆', '无溢', '下', '率志士', '三级', '盲目', '早通', '无处', '有胆有识', '服', '后历', '超攻', '披麻戴孝', '狎儿', '羽', '愚深', '绩', '虽处', '补常', '一路', '依理', '不入', '约好', '平子', '一眼', '动辄', '贱', '不屈于', '昔日', '全书', '隐于', '好像', '实', '异常', '总是', '多次', '壮士', '一天', '并没', '友好', '随使者', '言语', '从天', '不平', '辩白', '足迹', '静出', '致远', '终生', '宁愿', '聚兵', '永久性', '不堪吏职', '死活', '李谈', '留守', '不要', '忠诚', '接着', '简单', '不已', '以表', '仁慈', '十分', '本', '果真', '不久殿堂', '从弟', '仓皇', '曾巩', '其施', '颇知', '勘察', '从未', '以免', '详写', '季札', '威名', '专用', '安葬', '舍人', '历数', '已经', '顿觉', '向北', '过于', '真诚', '幸亏', '俘后', '蔡信', '少犯', '宦', '浇灌', '素以', '静江', '自上而下', '毕誡', '俭', '谋造', '是非曲直', '潜藏', '日夜', '原额', '韦曜', '干脆', '屈平', '先', '难测', '一连', '独宿', '另立', '片面', '约王', '荡平', '无限', '璐借', '永不', '热烈', '不当', '道济', '湖州', '因公', '圆满', '誉名', '延生性', '好德', '赵同', '尽心', '有为', '不甘', '最终', '护边', '相府', '同出', '康熙', '以烦', '历仕', '何为', '尽成', '杨慎', '堤防', '同朝人', '契丹', '京口', '士夺', '因罪', '善学', '带头', '罢官', '再', '邑令', '贬', '从狱', '邹容', '等一起', '必将', '诏嘉', '缨出', '惹得', '当久', '只顾', '以前', '并杀', '一般说来', '一度', '从荣', '一概', '全权', '讥讽', '诏文', '相传', '豪奸', '柳璨', '旧', '暂且', '年成', '五谷丰登', '说愿', '庸生', '生动', '平叛', '当北', '冷静', '甫', '风力', '晏', '为免', '就近', '均', '梁郁告', '看上去', '不慎', '丰厚', '不恭', '并不', '宁可', '始', '人丁', '如数', '初', '不得已', '至今', '匆忙', '明言', '主射者', '都督', '于是', '穰', '尖锐', '公正', '表面', '不失', '妒忌', '高颎', '尽占', '同民', '亦', '不计', '封国', '从于', '互不', '不改其乐', '必引', '迄今', '直升', '极少', '无师', '谳司', '绕山', '驱', '正面', '附权', '灵活', '毅然', '正趴', '齐', '尚野', '死', '尤其是', '沉到', '连年', '似乎', '唯独', '释疑', '广纳', '常常', '紫诰', '只怕', '安同', '深谙', '随人俯仰', '纵兵', '何武', '顺服', '可谓', '随西', '锯谷', '周密', '邑人', '优诏', '不少', '齐人', '瘦风', '举家', '这就', '不端', '细心', '祠', '授于', '不幸', '率鲁军', '确', '晓', '有的时候', '以后', '除孝', '匠人', '兵事', '饼供', '如夜', '并', '小心', '实为', '绝臣', '短于', '盖聂', '仲', '多日', '严历', '诸罗', '尽享', '典故', '叮', '品鉴', '臣下', '有主', '全交', '不明', '下任', '勘定', '名冠于', '涉猎', '曾弃官', '有识', '元朝', '怎', '轻', '永和', '真珈', '总管', '席上', '无端', '高君', '苏峻', '一石', '受劾', '狂妄', '分条', '雄厚', '聪慧', '独步', '因患', '陈寿', '一样', '特别', '任意', '相如', '横', '荒而', '在一起', '有望', '公然', '随军', '笃', '毫无疑问', '已然', '不如', '顺帝', '多方', '论功', '轻罚', '镇', '一五一十', '俗以', '有容乃大', '石氏', '不露', '差为', '只须', '翻船', '原先', '即将', '一点', '早熟', '其母', '从前', '上下', '真的', '急报', '赤诚', '督部', '见解', '名贵', '谪戍', '时贬', '死士', '近乎', '先议', '必', '片刻', '讨贼', '因帝', '酌情', '险易', '谆谆', '流泪', '正言', '日日', '合击', '董晋', '兵败', '进而', '只有', '羊枯', '正确', '罪张', '督都', '或宽', '诏物', '此文', '不息', '谢毅', '才气', '直', '以明', '范祥', '一个', '昭昭', '连坐', '泾师', '辛弃疾', '重新', '随三', '极言', '苻坚', '浑然', '谢晦', '绝不', '基本', '杜悰', '一下', '冗员', '从不曾', '仍率', '互相', '之所以', '累计', '过早', '不屈', '狄城', '重情', '睢阳', '广涉', '长时间', '轻易', '假意', '众多', '番人', '兼', '敏感', '幼弟', '全军', '近世', '奸险', '储将', '稳重', '那时候', '甚至', '悉数', '高权', '瑶族', '或许', '依旧', '设宴', '咬施', '雍阁', '至死', '反面', '稽', '周兴', '一再', '自由', '突然', '整兵', '阴历', '当初', '雄师', '萧渼', '随父', '骤', '自告奋勇', '一天到晚', '极其', '弘', '妄言', '宏阔', '浮官', '洪湛', '长久', '真心', '齐映', '接连', '当益', '潮州', '如杜', '责问', '恣意', '争先', '跪求', '敏锐', '严格', '为子', '直接', '安西行省', '平价', '绝食', '婺州', '远近', '寒朗', '病死', '无礼索', '乳母', '沈链', '应隐', '以勇', '据守', '人为', '时辞', '苦心', '全心', '尊老', '曹爽', '有声', '率先', '联合', '一边', '以度', '勉实', '有效', '刚直', '不由', '大大', '朱序', '实质上', '好', '在地', '刚者', '避权', '强烈', '显然', '一举', '偶尔', '羞愧', '连连', '任官', '恒山', '就是', '是否', '惠', '通常', '空费', '居室', '腹射', '尽职尽责', '一战', '连夜', '据说', '自小', '祖称', '多识', '后辈', '孝极', '例证', '节度', '拦路', '伟岸', '客观', '与人无争', '上奏书', '才', '器量', '庞籍', '并未', '分兵', '执意', '前后', '从旁', '从文', '著述', '赵范', '不怎么', '全盘', '高桢', '笔法', '萧懿', '救济', '佛寺', '谋略', '一定', '易', '终致', '频繁', '恍然', '堕马', '不相', '终于', '恩威', '从商', '次', '大呼', '大破', '空府', '另外', '棺椁', '笔墨', '事出', '当众', '更', '高宗', '以入', '稍', '立言', '因事', '早就', '无书', '祖援', '不至于', '灾期', '先归', '箭弩', '混同', '沈', '诗歌', '不善于', '初年', '严嵩', '从轻发落', '谱成', '直到', '此前', '威逼', '尽早', '从车', '时有', '道旁', '一步步', '喜', '独立性', '加爵', '永昌', '临终', '严正', '继位', '任成', '元昊', '周昌', '亲慰', '之立', '不图', '延缓', '仪说', '驱马', '仍', '李绅', '多种', '同时', '羊祜', '不免', '低调', '恩义', '用心', '未能', '多渠道', '还要', '邓尉', '不可避免', '隐隐', '充任', '奋勇', '衣衫', '终自', '由表及里', '忠烈', '除草', '其实', '加紧', '历迁', '好好', '严郢', '白镕', '老', '颈联', '尚未', '以众', '韦珍德', '重责', '废黜', '据理力争', '以备', '当下', '或', '牢牢', '朝政', '花销', '高', '下列', '增设', '从容镇', '主动', '授崔', '迫使区', '随主', '汴京', '岳飞', '先后', '何栗', '发愤', '确实', '苦;', '顿', '元兵', '懋', '恐惧', '当元', '可', '蒙混', '一起', '第后', '或暗', '逐级', '断狱', '畏罪', '蒲氏', '那天', '曾樱', '进一步', '不光', '只得', '苏辙', '多方面', '随同', '古来', '深', '当即', '特定', '此地', '迟眺', '曾经', '齿', '安然', '尤为', '顽强', '不仅仅', '那段', '仍旧', '渡航', '那样', '无辜', '稍微', '上级', '当庭', '到时', '难以', '大肆', '何胤', '辛弃', '随即', '仍愿', '欣然', '驼', '喜处', '疲惫', '治亊', '佀钟', '龚茂', '一味', '私下', '过分', '霸者', '有力', '舍舟', '皇上', '两', '自责', '根本', '高雅', '重信', '一生', '重加', '因位', '日趋', '连用', '暗中', '隐能', '规规矩矩', '一道', '多数', '单镒', '为学', '因交', '归山', '谯周', '兵败于', '爵', '好比', '新', '游时', '远', '相当', '相州', '默默', '破格', '僭越', '赶快', '谨慎', '起初', '难当', '追査', '永乐', '不满而立之年', '不轻率', '面对面', '超逸', '精忠', '吾粲', '善', '各', '尽可能', '如此', '大声', '宗望', '特地', '重在', '时常', '全境', '悲不能语', '疲倦', '睿智', '妃', '然后', '继之', '祖配', '远大', '密州', '勘谢', '贼兵', '紧急', '自视甚高', '明代', '按时', '临水', '劝勉', '刚毅', '瞋目', '但是', '惹恼', '借独', '无饭', '给到', '塘兼', '时刻', '其时', '从不', '前秦', '专门', '快速', '富裕', '恒', '祖予', '火速', '正宫', '尽快', '慨然', '凿', '略', '周苛', '纵横', '全', '先帝', '出色', '许氏', '亲身', '俯', '捻军', '姑且', '很', '没法', '解衣', '主要', '相宁', '索性', '寡', '湖心亭', '不善', '周洪', '银税', '能率', '幸亏子', '郑岳', '只不过', '果敢', '自比', '慈', '无论罪', '微山', '立刻', '谦委', '过往', '杨洪', '谏止', '孙思邈', '犹', '而已', '责备', '即源于', '以死相劝', '未必', '当然', '首联', '不得极', '勉强', '官们', '再三', '大都', '未入', '一气之下', '不只', '险峻', '指遭', '特别是', '怪罪', '奉诏', '耐心', '合理', '正常', '有心', '自然', '杜杲', '诸如', '从而', '习练', '刚正', '安器', '沿途', '因雨', '早日', '致力', '有时', '继昌', '严酷', '无论是', '奏表', '哪怕', '木讷', '郡举', '横道', '全部', '严谨', '刻苦', '当场', '颇', '将士们', '不久', '乱', '全身心', '从没', '竟', '朱震', '连射', '皇帝', '急忙', '一节', '妾告', '坦诚', '更击', '郑玄', '从谏如流', '势必', '突然间', '临朝', '当皇', '蓬门', '天天', '重义', '各种', '无意于', '谦虚', '明快', '因此', '狂热', '公护', '轻下', '帝力', '因顺', '将', '史弥', '贪士', '单纯', '侯琎', '这才', '极论', '尽心尽力', '胡乱', '翟光', '辗转', '安心', '没有', '堤堰', '佛嵩', '财', '待守', '至门', '平级', '幼年', '鉴', '刘表', '赈济', '一则', '活', '不和', '依然', '还家', '凝深', '恤', '极致', '一律', '以便', '燃柴', '辨正', '凭靠', '公有', '逐条', '程瑀', '稍稍', '幸好', '逐月', '说恩', '唆', '合', '隐藩', '凝正', '赞叹', '赐', '后', '病症', '且', '任性', '追道', '果园', '诏征', '刚', '特', '错误', '就地', '祥符', '自幼', '那', '一路上', '不慕', '必定', '正好', '天', '可比', '属大器', '章惇', '平时', '认真', '妾室', '轻视', '打理', '看来', '重节', '早前', '安礼', '尽日', '悄然', '照样', '亲点', '正御', '深切', '奏事', '余良', '却', '愤愤', '严加', '何景', '随意', '顺势', '畏懦', '重耳翻', '永福', '翟銮', '数挫', '提示', '自罚', '不曾', '可悲', '阴嵩', '自动', '朝代', '现代', '厘负', '侧目', '历抵', '有使者', '就说', '代', '遇春', '未', '尽情', '后官', '总体而言', '简便', '以致', '派使者', '诸家学', '庚午', '完全', '屯田', '浙东', '对母', '永远', '藉田', '高适', '同等', '好学', '淳', '不用', '卖文', '惠王', '最后', '可是', '酣战', '深刻', '勤俭', '詹事', '这时', '誓死', '率性', '方正', '派桑', '高拱', '匆匆', '多年', '如僧', '无疑', '伏击', '叫声', '客观上', '毕竟', '丞相', '晓崇', '梓人', '及时', '无奈', '落后', '简要', '反', '拜任', '勤劳', '永明', '诸将', '素来', '从面', '越来越', '比作', '不得不', '挫', '因告', '不过', '整天', '兵索', '即便', '反而', '可指', '遇则', '进刚', '墓地', '无钱', '腹较', '进谏', '即可', '后来', '有时候', '兵法', '久掌', '玄宗', '种', '沈恪', '整整', '郎先', '浅近', '大志', '巧设', '专', '细密', '据实', '俯瞰', '并流', '爱民', '直言不讳', '屈突', '率兵', '大船', '有善于', '单单', '一番', '综合', '实地', '马旦', '历来', '细柳', '闲步', '冒雨', '长期', '恰当', '高见', '尚书省', '不是', '韦放', '晁', '兖', '不够', '禹', '兵部', '后常', '不论', '五经', '拔拳', '其家', '擅离', '低劣', '详细', '节令', '实在', '择人', '郡人', '义举', '周起', '依法', '景仁', '前', '层层', '逐年', '都', '曾使金', '公开', '厚加', '佩服', '普通', '仲遵', '通有', '何武仁义', '常', '常人', '遣出', '晟本', '下诏', '约略', '隆重', '品性', '应举', '决', '连续', '侯放', '扇薛', '深明', '迁往', '明晓', '叔段', '孝公', '暗地里', '横征暴敛', '自行', '精于', '否则', '范增', '鉴湖', '虑事', '恐怕', '尽忠', '而是', '居然', '也许', '绍兴', '仗义', '早立', '夜夜', '偷偷', '与卿', '具体', '纯朴', '终因', '刘玉', '文武', '以事', '诏令', '勇敢', '敏捷', '照旧', '康绚', '才华横溢', '急着', '想方设法', '吾子', '持续', '冒犯', '一般', '之后', '明南', '奋力', '竭力', '瘙痒难忍', '擂鼓', '慌乱', '连忙', '伍文定', '帝误', '如征', '节妇', '崩', '诚恳', '愈加', '迫切', '广布', '宽大', '彭越', '不得', '抢先', '坚毅', '以史', '临时', '敢作敢为', '黯然', '视江', '难成', '玺', '数日不觉', '如刚', '难道', '余玠镇', '再次', '一经', '杖责', '老者', '随缘', '明', '擅权', '甚', '明知', '果如', '凡', '年轻', '三年', '终', '尚书', '简洁', '互', '所幸', '最早', '诸峰', '严地', '基本上', '由此', '平安', '精心', '看后', '杨绘', '顶住', '辰', '一清二楚', '夺情', '不妨', '不', '其后', '玺传', '必成', '侯植', '不知', '祖当', '孙坚战', '深深', '不时', '以守', '每每', '日渐', '加之', '单马率', '极恭', '廉洁', '叛贼', '广为', '叔母', '愈', '轻重', '实征', '在', '卓异', '横扫', '精细', '自修', '奏', '严辞', '着重', '日便', '含羞', '由宫', '冒险', '在先', '无且', '墨子', '先士', '让伍', '安从', '就', '最', '普遍', '巧妙', '自度', '合力', '惩凶', '程震', '相对', '谋事', '就是说', '每年', '孤独', '亲', '因灾', '何谪', '上疏', '狂怒', '在所不惜', '无锡县', '虚实', '四季', '倾巢', '与其说', '过后', '九卿', '卓越', '可恕', '漫步', '短暂', '卞彪', '林尉', '借机', '一劫', '始终', '免遭', '以气', '四面八方', '最初', '濮州', '据理', '艰难', '妥善', '寡人', '难怪', '主', '仇鸾', '坚定', '大为', '进犯', '一', '不觉', '遥远', '原本', '终使', '击', '邓禹', '屡', '借古', '于', '晏殊', '才干', '无需', '相继', '独子', '只用', '吏部', '稳定', '整形', '不知不觉', '大发', '非', '无为', '入京', '左右', '毕士', '殁后', '恒屡', '直名', '壮', '不如说', '狭窄', '咬指', '猛', '每月', '惟', '张紞', '老臣', '深怀', '赐归', '刚刚', '黄福', '细中', '随后', '方', '深以', '稷为', '险遭', '公平', '轻狂', '廉约', '时任', '豁然', '初战', '并且', '强', '诬奏', '律法', '清平', '常责', '徒留', '及早', '连日', '如实', '预先', '平实', '尉迥', '先礼后兵', '越级', '悲伤', '此法', '频频', '共同', '冰晶', '正式', '帅边', '提学', '终遭', '颇顺', '明著', '曾', '智谋', '以上', '共', '趁机', '帅营', '后娶', '继而', '宗', '不守', '并门', '先世', '怒视', '终招', '亲切', '宾天', '永放', '持刀', '还有', '何', '立志', '必然', '虚席', '越发', '仇钺', '急切', '常同', '断然', '固然', '梁军', '恤百姓', '恳劝', '有诈', '仿佛', '宣宗派', '右拾', '虽谐', '从师', '先秦', '早先', '难得', '择善而从', '大力', '庸', '从容', '宁肯', '迂回', '助长', '一手', '逐个', '铎', '而后', '随客', '以渡', '自我', '主客', '虽历', '明察', '派往', '普于', '无业', '丝毫', '一一', '和平', '谄媚权', '均分', '再度', '反北', '也', '名实', '有理有据', '原任', '泗州', '寒冷', '酒邢', '绿', '多久', '真是', '准确', '五升', '坚决', '非亲征', '一文', '与子', '高谷', '一元', '盛情', '忙着', '世世代代', '宽以待人', '俯伏', '堆满', '正在', '卧', '蓄意', '成年', '轻财', '直言', '跟着', '安善', '晚辈', '长年', '随便', '裁处', '身出', '反过来', '典雅', '重', '重重', '如期', '颜氏', '蕃人', '兵随', '任情', '互为', '事实上', '切实', '筑坛', '假', '悄悄', '自', '敬', '随严', '顾越', '诚', '的确', '赐爵', '薛胄', '不然', '以行', '亲手', '为什么', '运费', '义正辞严', '原物', '灵隽', '秉公', '屡次', '即事', '多处', '持剑', '囚车', '无论', '真定', '借指', '典章', '周勃', '赠谥', '竞相']
# print(len(a))
#
# sp = len(a)/15
#
# for i in range(15):
#     print(a[int(i*sp):int((i+1)*sp)])

#-*-encoding=utf8-*-
from stanfordcorenlp import StanfordCoreNLP


nlp = StanfordCoreNLP(r'/disk2/huanggj/stanford-corenlp-4.4.0', lang='en',timeout=20)
sentence="hello world , my name is goumeng"
print(nlp.word_tokenize(sentence))
print(nlp.pos_tag(sentence))
print(nlp.ner(sentence))
print(len(nlp.ner(sentence)))
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))
nlp.close()