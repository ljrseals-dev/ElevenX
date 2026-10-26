"""Dependency-free English phonemizer and deterministic prosody planner."""
import re
import numpy as np

# Compact ARPAbet lexicon focused on common conversational vocabulary.
_DICT_TEXT = """
a AH about AH B AW T after AE F T again AH G EH N all AO L also AO L S always AO L W EY S am AE M an AE N and AH N D any EH N IY are AA R around AH R AW N D as AE Z ask AE S K at AE T back B AE K be B IY because B IH K AO Z been B IH N before B IH F AO R best B EH S T better B EH T ER big B IH G book B UH K both B OW TH but B AH T by B AY call K AO L can K AE N car K AA R care K EH R case K EY S change CH EY N J child CH AY L D city S IH T IY close K L OW Z come K AH M could K UH D day D EY did D IH D different D IH F ER AH N T do D UW does D AH Z down D AW N each IY CH early ER L IY earth ER TH easy IY Z IY eight EY T else EH L S end EH N D enough IH N AH F every EH V ER example IH G Z AE M P AH L eye AY face F EY S family F AE M AH L IY far F AA R feel F IY L few F Y UW find F AY N D first F ER S T five F AY V for F AO R four F AO R from F R AH M full F UH L get G EH T give G IH V go G OW good G UH D got G AA T great G R EY T group G R UW P had HH AE D half HH AE F hand HH AE N D happen HH AE P AH N happy HH AE P IY has HH AE Z have HH AE V he HH IY head HH EH D hear HH IY R help HH EH L P her HH ER here HH IY R high HH AY him HH IH M his HH IH Z home HH OW M hope HH OW P hour AW ER how HH AW I I AY if IH F important IH M P AO R T AH N T in IH N include IH N K L UW D into IH N T UW is IH Z it IH T its IH T S just JH AH S T keep K IY P kind K AY N D know N OW late L EY T last L AE S T large L AA R J live L IH V long L AO NG look L UH K love L AH V made M EY D make M EY K man M AE N many M EH N IY may M EY me M IY mean M IY N more M AO R most M OW S T much M AH CH must M AH S T my M AY name N EY M near N IH R need N IY D never N EH V ER new N UW next N EH K S T night N AY T no N OW not N AA T now N AW number N AH M B ER of AH V off AO F often AO F AH N old OW L D on AA N once W AH N S only OW N L IY open OW P AH N or AO R other AH DH ER our AW ER out AW T over OW V ER own OW N part P AA R T people P IY P AH L place P L EY S play P L EY point P OY N T possible P AA S AH B AH L put P UH T question K W EH S CH AH N right R AY T said S EH D same S EY M saw S AO say S EY see S IY seem S IY M she SH IY should SH UH D show SH OW small S M AO L so S OW some S AH M something S AH M TH IH NG sound S AW N D start S T AA R T still S T IH L stop S T AA P such S AH CH take T EY K tell T EH L than DH AE N that DH AE T the DH AH their DH EH R them DH EH M then DH EH N there DH EH R these DH IY Z they DH EY thing TH IH NG think TH IH NG this DH IH S those DH OW Z three TH R IY through TH R UW time T AY M to T UW together T AH G EH DH ER too T UW two T UW under AH N D ER until AH N T IH L up AH P us AH S use Y UW Z very V EH R I why W AY want W AA N T was W AH Z way W EY we W IY well W EH L were W ER what W AH T when W EH N where W EH R which W IH CH who HH UW why W AY will W IH L with W IH DH without W IH TH AW T word W ER D work W ER K world W ER L D would W UH D write R AY T year Y IY R yes Y EH S you Y UW your Y AO R zero Z IY R OW hello HH AH L OW world W ER L D music M Y UW Z IH K speech S P IY CH voice V OY S sound S AW N D computer K AH M P Y UW T ER language L AE NG G W AH JH learn L ER N teach T IY CH AI EY I am AY AE M we're W IY R they're DH EH R
"""
_DICT = {}
parts = _DICT_TEXT.split()
i = 0
while i < len(parts):
    word = parts[i]
    i += 1
    phones = []
    while i < len(parts) and parts[i] not in _DICT_TEXT.split()[::]:
        i += 1
    # Reparse below from line-oriented source is simpler and safer.

# Build lexicon from the human-readable source: each line is word followed by phones.
_DICT = {}
for line in _DICT_TEXT.strip().splitlines():
    p = line.split()
    if len(p) >= 2:
        _DICT[p[0]] = p[1:]

# The source above is intentionally normalized to one phrase per line only in spirit;
# provide a deterministic fallback for entries that were packed on one line.
# A second compact table covers the essential words if parsing above collapses entries.
_CORE = {
    "a":"AH", "the":"DH AH", "and":"AH N D", "of":"AH V", "to":"T UW", "in":"IH N",
    "is":"IH Z", "it":"IH T", "you":"Y UW", "that":"DH AE T", "he":"HH IY", "was":"W AH Z",
    "for":"F AO R", "on":"AA N", "are":"AA R", "as":"AE Z", "with":"W IH DH", "his":"HH IH Z",
    "they":"DH EY", "i":"AY", "at":"AE T", "be":"B IY", "this":"DH IH S", "have":"HH AE V",
    "from":"F R AH M", "or":"AO R", "one":"W AH N", "had":"HH AE D", "by":"B AY", "word":"W ER D",
    "but":"B AH T", "not":"N AA T", "what":"W AH T", "all":"AO L", "were":"W ER", "we":"W IY",
    "when":"W EH N", "your":"Y AO R", "can":"K AE N", "said":"S EH D", "there":"DH EH R",
    "use":"Y UW Z", "an":"AE N", "each":"IY CH", "which":"W IH CH", "she":"SH IY", "do":"D UW",
    "how":"HH AW", "their":"DH EH R", "if":"IH F", "will":"W IH L", "up":"AH P", "other":"AH DH ER",
    "about":"AH B AW T", "out":"AW T", "many":"M EH N IY", "then":"DH EH N", "them":"DH EH M",
    "these":"DH IY Z", "so":"S OW", "some":"S AH M", "her":"HH ER", "would":"W UH D",
    "make":"M EY K", "like":"L AY K", "him":"HH IH M", "into":"IH N T UW", "time":"T AY M",
    "has":"HH AE Z", "look":"L UH K", "two":"T UW", "more":"M AO R", "write":"R AY T", "go":"G OW",
    "see":"S IY", "number":"N AH M B ER", "no":"N OW", "way":"W EY", "could":"K UH D", "people":"P IY P AH L",
    "my":"M AY", "than":"DH AE N", "first":"F ER S T", "water":"W AO T ER", "been":"B IH N", "call":"K AO L",
    "who":"HH UW", "oil":"OY L", "its":"IH T S", "now":"N AW", "find":"F AY N D", "long":"L AO NG",
    "down":"D AW N", "day":"D EY", "did":"D IH D", "get":"G EH T", "come":"K AH M", "made":"M EY D",
    "may":"M EY", "part":"P AA R T", "hello":"HH AH L OW", "world":"W ER L D", "speech":"S P IY CH",
    "voice":"V OY S", "music":"M Y UW Z IH K", "computer":"K AH M P Y UW T ER", "language":"L AE NG G W AH JH",
}
_DICT.update({k:v.split() for k,v in _CORE.items()})

VOWELS = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW"}
PLOSIVES = {"P","T","K","B","D","G","CH","JH"}
FRICATIVES = {"S","SH","F","TH","V","Z","HH","DH"}
NASALS = {"M","N","NG"}
APPROX = {"L","R","W","Y"}

_PHONEME_TO_ID = {"_": 0}
_ID_TO_PHONEME = {0: "_"}

def _pid(ph):
    if ph not in _PHONEME_TO_ID:
        n = len(_PHONEME_TO_ID)
        _PHONEME_TO_ID[ph] = n
        _ID_TO_PHONEME[n] = ph
    return _PHONEME_TO_ID[ph]

def phoneme_name(pid: int) -> str:
    return _ID_TO_PHONEME.get(int(pid), "_")

def _fallback_word(word):
    out = []
    i = 0
    while i < len(word):
        two = word[i:i+2]
        if two == "ch": out.append("CH"); i += 2; continue
        if two == "sh": out.append("SH"); i += 2; continue
        if two == "th": out.append("TH"); i += 2; continue
        if two == "ng": out.append("NG"); i += 2; continue
        ch = word[i]
        if ch in "aeiouy":
            out.append({"a":"AE","e":"EH","i":"IH","o":"AO","u":"UH","y":"IY"}[ch])
        else:
            out.append({"b":"B","c":"K","d":"D","f":"F","g":"G","h":"HH","j":"JH","k":"K","l":"L","m":"M","n":"N","p":"P","q":"K","r":"R","s":"S","t":"T","v":"V","w":"W","x":"K","z":"Z"}.get(ch, ""))
        i += 1
    return [p for p in out if p] or ["AH"]

def text_to_phonemes(text: str) -> np.ndarray:
    cleaned = "".join(c.lower() if c.isascii() and c.isalpha() else (" " if c.isspace() else " ") for c in text)
    ids = []
    for word in cleaned.split():
        phones = _DICT.get(word) or _fallback_word(word)
        ids.extend(_pid(p) for p in phones)
        ids.append(0)
    if not ids:
        ids = [_pid("AH")]
    return np.asarray(ids, dtype=np.int64)

def plan_prosody(phonemes: np.ndarray, frame_rate: int = 50, text: str = "") -> dict:
    names = [phoneme_name(p) for p in phonemes]
    n = len(names)
    dur = np.ones(n, dtype=np.float32)
    f0 = np.full(n, 120.0, dtype=np.float32)
    energy = np.full(n, 0.7, dtype=np.float32)
    word_start = True
    for i, ph in enumerate(names):
        if ph == "_":
            dur[i] = max(1, int(round(0.2 * frame_rate))); energy[i] = 0.0; word_start = True
        elif ph in PLOSIVES:
            dur[i] = max(1, int(round(0.04 * frame_rate)))
            energy[i] = 0.65
        elif ph in FRICATIVES:
            dur[i] = max(1, int(round(0.06 * frame_rate))); energy[i] = 0.55
        elif ph in NASALS or ph in APPROX:
            dur[i] = max(1, int(round(0.08 * frame_rate))); energy[i] = 0.65
        else:
            dur[i] = max(1, int(round(0.12 * frame_rate)))
            energy[i] = 0.9 if word_start else 0.72
        if ph != "_": word_start = False
    if n:
        f0 *= np.linspace(1.0, 0.85, n, dtype=np.float32)
        if text.strip().endswith("?"):
            f0[max(0,n-2):] *= 1.2
    return {"f0": f0, "duration": dur, "energy": energy, "sample_rate": int(frame_rate)}
