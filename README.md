# CHEAT
The official repository of the paper CHEAT: A Large-scale Dataset for Detecting CHatGPT-writtEn AbsTracts

The powerful ability of ChatGPT has caused widespread concern in the academic community. Malicious users could synthesize dummy academic content through ChatGPT, which is extremely harmful to academic rigor and originality. The need to develop ChatGPT-written content detection algorithms call for large-scale datasets. We initially investigate the possible negative impact of ChatGPT on academia, and present a large-scale CHatGPT-writtEn AbsTract dataset (CHEAT) to support the development of detection algorithms. In particular, the ChatGPT-written abstract dataset contains 35,304 synthetic abstracts, with $Generation$, $Polish$, and $Fusion$ as prominent representatives. Based on these data, we perform a thorough analysis of the existing text synthesis detection algorithms. We show that ChatGPT-written abstracts are detectable with well-trained detectors, while the detection difficulty increases with human involvement.

## Dataset
In the data directory there are the files ieee-{init, generation, polish, fusion}.{jsonl, xlsx}. These contain the original (human) input in the init files, the first pass generation by ChatGPT in generation, the ChatGPT refactored versions in polish, and the human/ChatGPT hybrids in fusion.
