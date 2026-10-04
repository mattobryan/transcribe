import json,re,collections
from wordfreq import zipf_frequency, top_n_list
from lingua import Language, LanguageDetectorBuilder
d=LanguageDetectorBuilder.from_languages(Language.ENGLISH,Language.SWAHILI).build()
import sys
t=json.load(open(sys.argv[1]))['text']          # an ElevenLabs Scribe transcript JSON
toks=re.findall(r"[A-Za-z]+",t)
cnt=collections.Counter(x.lower() for x in toks)
cap=collections.Counter()
for m in re.finditer(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]+)\b",t): cap[m.group(1).lower()]+=1
NAMES={'mwaka','nyale','mbagathi','kenyatta','tizoo','patience'}
sw=set()
for w,c in cnt.items():
    sc=next(x.value for x in d.compute_language_confidence_values(w) if x.language==Language.SWAHILI)
    if sc>=0.9 and zipf_frequency(w,'en')<3.3 and w not in NAMES and cap[w]<max(1,0.5*c) and len(w)>=3: sw.add(w)
EXTRA="""na ni kwa za pia tu ama ndo ndio ndiyo si hapana sawa asante pole karibu sana sasa hapa huko kule pale hii hiyo hizi hizo huyu huyo hawa hao yule wale ile ule kila kama lakini au kwamba bado tena kabisa leo jana kesho wakati mara moja mbili tatu nne tano sita saba nane tisa kumi mimi wewe yeye sisi nyinyi wao nimeshukuru mama baba dada kaka rafiki mwanangu watoto mtoto mtu watu kazi hospitali dawa daktari mgonjwa wagonjwa muuguzi nesi shida tatizo hali sawasawa vizuri mbaya nzuri kubwa ndogo mpya zamani sasa baadaye kwanza halafu kisha hivyo hivi vile kuna hakuna ana ina una ako uko aje wapi lini nani nini gani vipi mbona kwanini kwani eti yaani basi ndiyo labda pengine kweli kabisa ngoja njoo twende tuende nenda enda sema ona jua fanya pata leta peleka kula nywa lala amka keti simama subiri sikia sikiliza ongea ongezea hapo hivo sina acha nasina wambie siendi picha pesa nikae sione bade aone waseme ingine kukula""".split()
sw|=set(EXTRA)
en=[w for w in top_n_list('en',60000) if re.fullmatch(r"[a-z]{2,}",w)][:30000]
en=[w for w in en if w not in sw]
open('src/sw_words.txt','w').write(' '.join(sorted(sw))); open('src/en_words.txt','w').write(' '.join(sorted(en)))
print(len(sw),len(en),len(' '.join(en)))
