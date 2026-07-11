# A Comprehensive Survey of Multimodal Large Language Models: Architectures, Training Paradigms, Benchmark Evaluation, and Future Directions

Authors: ISSA ISSA RASHID  
Affiliation: School of Computer Science and Technology, Harbin Institute of Technology, Shenzhen, China  
Corresponding author: ISSA ISSA RASHID, 25sf51115@stu.hit.edu.cn

## Abstract

Multimodal large language models (MLLMs) extend the reasoning, instruction-following, and generation abilities of large language models to inputs such as images, diagrams, video frames, optical text, charts, and visually grounded user instructions. Their rapid development has changed the research landscape of vision-language learning: earlier systems optimized task-specific objectives for retrieval, captioning, or visual question answering, whereas modern MLLMs attempt to serve as general interfaces between visual perception and language-based reasoning. This survey reviews the technical foundations, architecture families, training paradigms, benchmark practices, safety issues, and reproducibility requirements of MLLMs. The discussion is organized around four central questions: how visual information is encoded, how it is connected to language models, how multimodal alignment and instruction tuning are performed, and how current benchmarks measure perception, reasoning, hallucination, and expert-domain competence. Representative systems including CLIP, Flamingo, BLIP-2, LLaVA, Qwen-VL, InternVL, CogVLM, and GPT-4V are analyzed alongside earlier vision-language pretraining models and recent grounding-oriented MLLMs. The paper does not report new model-training experiments or new benchmark scores, because no controlled benchmark execution was performed. Instead, it provides a reproducible literature and taxonomy package containing verified references, structured model and benchmark tables, generated figures, formatting scripts, and clear code-availability information. The main conclusion is that MLLM progress should be evaluated through architecture, grounding reliability, hallucination behavior, safety, benchmark transparency, and reproducibility rather than through isolated leaderboard scores.

## Keywords

multimodal large language models, vision-language learning, instruction tuning, benchmark evaluation, hallucination, reproducibility

## 1. Introduction

Large language models have become general-purpose interfaces for natural-language reasoning, code generation, knowledge synthesis, and interactive problem solving. Their success depends on scaling model capacity, training data, and transformer-based sequence modeling [1], together with learning paradigms that allow a single model to generalize across many tasks from prompts or few examples [2]. However, real-world intelligence is not limited to text. Humans interpret diagrams, images, equations, maps, user interfaces, videos, charts, and physical scenes in combination with language. A model that can only process text is therefore structurally limited in domains such as scientific education, robotics, document understanding, medical image interpretation, visual programming, accessibility, and human-computer interaction.

Multimodal large language models address this limitation by coupling visual encoders, multimodal connectors, and language models. They are not merely image classifiers with text output. A capable MLLM must map visual input into a representation that a language model can reason over; preserve spatial, semantic, and symbolic details; follow natural-language instructions; avoid hallucinating nonexistent content; and maintain useful language ability after multimodal adaptation. These requirements create a complex design space. Some models use dual encoders for image-text representation learning, as in CLIP [3] and ALIGN [4]. Others use cross-attention layers, query transformers, projection modules, or visual expert modules to connect vision encoders with frozen or partially trainable language models [5] [6] [7]. A third family emphasizes instruction tuning with image-text dialogue data, as in LLaVA [8], InstructBLIP [9], MiniGPT-4 [10], and Qwen-VL [11].

The field has moved quickly because MLLMs inherit two sources of progress. From computer vision, they inherit stronger image encoders, richer pretraining corpora, and improved grounding methods. From language modeling, they inherit instruction-following behavior, chain-of-thought style reasoning, retrieval-augmented workflows, tool use, and deployment ecosystems. This combination creates impressive qualitative behavior, but it also introduces evaluation difficulties. A model may answer a question fluently while misreading the image, grounding text incorrectly, or relying on language priors. Benchmark performance can be sensitive to prompt templates, answer extraction, dataset leakage, and scoring rules. Consequently, survey work in this area must distinguish real measured evidence from demonstration claims.

This paper surveys MLLMs from the perspective of architecture, training, evaluation, and reproducibility. It focuses on models and benchmarks that have shaped the research conversation: CLIP, Flamingo, BLIP-2, InstructBLIP, LLaVA, MiniGPT-4, Qwen-VL, InternVL, CogVLM, and GPT-4V [12]. It also examines benchmark families including VQA [13], GQA [14], OK-VQA [15], ScienceQA [16], MME [17], MMBench [18], MM-Vet [19], MMMU [20], POPE [21], and MathVista [22]. No new benchmark scores are reported. Where benchmark names are discussed, the purpose is to explain what they evaluate and why they matter.

The contributions of this survey are fourfold. First, it provides a structured taxonomy of MLLM architectures, separating representation learning, connector-based adaptation, instruction-tuned systems, and visual-expert designs. Second, it compares training paradigms, including contrastive pretraining, generative captioning, bootstrapped alignment, visual instruction tuning, and safety alignment. Third, it analyzes benchmark categories and the practical risks of over-interpreting leaderboard numbers. Fourth, it provides a reproducible local package containing the paper draft, bibliography, taxonomy tables, and scripts for table generation and citation validation.

## 2. Related Work

Research on MLLMs builds on several earlier lines of work: transformer sequence modeling, vision-language representation learning, visual question answering, image captioning, and instruction-following language models. The transformer architecture introduced self-attention as a scalable mechanism for sequence modeling [1]. Large language models later demonstrated that scaling and broad pretraining can produce strong few-shot behavior across tasks [2]. These developments supplied the language backbone used by many modern MLLMs.

Vision-language representation learning provided the second foundation. Earlier transformer-based systems such as ViLBERT, LXMERT, UNITER, Oscar, VinVL, and OFA studied cross-modal pretraining before the current MLLM wave [23] [24] [25] [26] [27] [28]. CLIP trained image and text encoders with a contrastive objective over image-caption pairs, producing transferable open-vocabulary visual representations [3]. ALIGN scaled a similar idea with noisy image-text supervision and showed that large web-scale data could support robust cross-modal retrieval and classification [4]. These models are not full MLLMs in the current sense because they do not perform open-ended multimodal dialogue through a generative language model. Nevertheless, they established a practical strategy: align visual and textual representations using paired data, then reuse the learned representation for downstream tasks.

A second group of works moved from retrieval-style representation learning toward generative vision-language models. BLIP unified understanding and generation with bootstrapped captions and filtering [29]. Flamingo connected strong visual encoders to frozen language models with cross-attention layers and demonstrated few-shot learning over interleaved image, video, and text inputs [5]. BLIP-2 introduced a Querying Transformer, commonly called Q-Former, to bridge frozen image encoders and frozen language models efficiently [6]. These models are important because they show that full end-to-end retraining is not the only route to multimodal competence. Careful adapter or connector design can exploit powerful unimodal components while reducing training cost.

Instruction tuning then became central. InstructBLIP adapted the BLIP-2 design toward instruction following [9]. LLaVA showed that visual instruction data could turn a vision encoder and a language model into an open multimodal assistant [8]. MiniGPT-4 provided another compact alignment recipe using a frozen visual encoder and a Vicuna-style language model [10]. These works changed the evaluation focus from task-specific metrics to interactive behavior: the model must answer questions, describe scenes, reason over diagrams, and follow user intent.

Recent systems have expanded the design space. PaLI and PaLM-E illustrate multilingual and embodied multimodal scaling [30] [31]. Kosmos-1 and Kosmos-2 emphasize aligning perception with language and grounding multimodal language models to the world [32] [33]. Qwen-VL emphasizes abilities such as text reading and localization in addition to general visual dialogue [11]. mPLUG-Owl, Shikra, Ferret, VILA, Video-LLaVA, and Visual ChatGPT represent additional directions in modularization, referential dialogue, grounding, pretraining, video alignment, and tool-assisted visual interaction [34] [35] [36] [37] [38] [39]. InternVL studies scaling of vision foundation models and alignment for generic visual-linguistic tasks [40]. CogVLM uses a visual expert approach to integrate visual capability into pretrained language models while preserving language competence [7]. GPT-4V represents a closed frontier system described through a system card rather than full open model weights or training details [12]. The contrast between open and closed systems is now a major reproducibility issue: closed systems can be evaluated externally, but their training data, architecture, and alignment processes cannot be independently inspected.

Evaluation research has also evolved. Early visual question answering benchmarks such as VQA [13], GQA [14], and OK-VQA [15] tested specific combinations of perception, language, compositional reasoning, and external knowledge. TextVQA and ChartQA extended evaluation toward visual text reading and chart reasoning [41] [42]. ScienceQA introduced science-question reasoning with multimodal explanations [16]. Modern MLLM benchmarks such as MME, MMBench, MM-Vet, and MMMU attempt to measure broader capability dimensions [17] [18] [19] [20]. POPE focuses specifically on object hallucination [21], while MathVista evaluates mathematical reasoning in visual contexts [22]. This benchmark expansion reflects a key point: no single test can characterize an MLLM.

## 3. Theoretical Foundation

An MLLM can be viewed as a conditional sequence model that generates text given multimodal context. Let an image or set of images be denoted by \(x_v\), a textual prompt by \(x_t\), and the target textual response by \(y = (y_1, ..., y_T)\). A generative MLLM models

\[
p(y | x_v, x_t) = \prod_{i=1}^{T} p(y_i | y_{<i}, x_t, f_v(x_v)),
\]

where \(f_v\) is a visual encoder or visual representation module. The central design problem is how to transform \(f_v(x_v)\) into representations compatible with the token-based language model. Different MLLM families answer this problem differently.

In a dual-encoder contrastive model, an image encoder \(f_v\) and a text encoder \(f_t\) map inputs into a shared embedding space. Given paired examples \((x_v^i, x_t^i)\), a contrastive loss increases similarity for matched pairs and decreases similarity for mismatched pairs. This approach supports retrieval and open-vocabulary classification, but it does not by itself generate long-form answers. CLIP is the canonical example [3]. Its importance for MLLMs lies in the learned visual representation and image-text alignment, which can be reused in later systems.

Connector-based MLLMs add an interface between visual features and a language model. Let \(h_v = f_v(x_v)\) be a sequence or grid of visual features. A connector \(g\) maps visual features into a sequence of embeddings \(z_v = g(h_v)\) compatible with the language model token embedding space. The language model then receives a combined sequence such as \([z_v; e(x_t)]\), where \(e(x_t)\) denotes text token embeddings. The connector may be a linear projection, multilayer perceptron, cross-attention module, Q-Former, or more elaborate visual expert. Linear projections are simple and efficient, but they may bottleneck spatial and symbolic detail. Query-based connectors can compress visual features into a fixed number of learned queries, improving efficiency but potentially discarding fine-grained evidence. Cross-attention layers allow the language model to attend to visual states more flexibly, but they increase architectural complexity.

Instruction tuning changes the learning objective. Instead of only predicting captions or matching image-text pairs, the model is trained on examples of the form \((image, instruction, answer)\). The supervised objective is usually next-token prediction over the answer. This procedure aligns the model with conversational behavior and user intent. However, instruction data quality is critical. If the instruction data contains overconfident answers, weak grounding, or synthetic artifacts, the model may learn to respond fluently without adequate visual evidence. The risk is especially serious for hallucination, where a model mentions objects, text, or relations that are not present in the image.

The theoretical tension in MLLMs is therefore between compression and grounding. Visual input is high-dimensional and often spatially structured. Language models operate over token sequences and learn strong textual priors. A model must compress images enough for efficient reasoning but not so aggressively that it loses fine details. It must exploit language knowledge without allowing language priors to override visual evidence. It must learn instruction-following behavior without treating every prompt as an invitation to invent plausible content. Many current limitations follow from this tension.

## 4. Methodology

### 4.1 Problem Definition

This paper studies MLLMs as systems that accept one or more visual inputs and natural-language prompts, then produce natural-language outputs that are intended to be grounded in the visual input. The survey asks four research questions:

1. What architecture patterns dominate current MLLMs?
2. What training and alignment paradigms are used to connect vision and language?
3. What benchmark families evaluate perception, reasoning, hallucination, and expert-domain ability?
4. What reproducibility practices are necessary for trustworthy MLLM research?

The paper is a literature survey and taxonomy study. It does not train a model, fine-tune a model, evaluate a benchmark, or report new scores. This choice is deliberate: the available local project does not include model weights, benchmark datasets, compute logs, or executed experiment outputs. Reporting new numbers without execution would violate reproducibility requirements.

### 4.2 Literature Selection

The paper prioritizes real references from major venues and widely cited technical reports or arXiv preprints that shaped the MLLM field. The selected works cover foundational transformers, large language model scaling, contrastive vision-language learning, generative vision-language pretraining, instruction-tuned MLLMs, closed frontier multimodal systems, and multimodal benchmarks. The bibliography is stored in `paper/references.bib`, and citation keys in the manuscript can be checked with `scripts/validate_references.py`.

The selection is not exhaustive. MLLMs are a rapidly moving field, and many new models appear as technical reports before peer review. For that reason, this survey emphasizes stable conceptual categories rather than a complete leaderboard. Future extensions should add references only after verifying bibliographic details and should distinguish peer-reviewed papers from preprints and system cards.

### 4.3 Taxonomy Construction

Two taxonomy tables are included. The model taxonomy records representative systems, publication year, architecture pattern, training or alignment paradigm, and notable role. The benchmark taxonomy records benchmark year, primary focus, and typical use in MLLM evaluation. These tables are stored as CSV files in `paper/tables/` and can be regenerated as Markdown tables by running:

```bash
python scripts/generate_tables.py
```

This design makes the survey easier to audit. Instead of manually editing tables in the paper, future work can update structured CSV files and regenerate consistent output.

### 4.4 Implementation Details

Because this is a survey paper, there are no model-training hyperparameters such as optimizer, learning rate, batch size, epochs, or weight decay. If future original experiments are added, those fields must be specified exactly. For the current reproducibility package, the implementation details are:

- Programming language: Python 3.11
- Required package: PyYAML for configuration compatibility
- Scripts: `generate_tables.py` and `validate_references.py`
- Hardware: CPU-only execution is sufficient for the included scripts
- Random seed: not applicable because no stochastic experiment is executed
- Software environment: specified in `requirements.txt` and `environment.yml`

## 5. Architecture Taxonomy

### 5.1 Contrastive Representation Models

Contrastive vision-language models learn aligned image and text spaces. CLIP is the most influential example [3]. It trains image and text encoders so that matching image-caption pairs have high similarity and mismatched pairs have low similarity. This objective enables zero-shot image classification by comparing an image embedding to text prompts representing class names. ALIGN follows a related strategy at larger noisy-data scale [4].

The strength of contrastive models is transferability. They produce visual representations that can support classification, retrieval, and downstream multimodal systems. Their weakness is generative limitation. A dual encoder does not naturally produce a paragraph-length answer, a dialogue response, or a step-by-step explanation. For that reason, contrastive models often serve as components inside broader MLLMs rather than complete assistants.

### 5.2 Generative Vision-Language Pretraining

BLIP introduced bootstrapping methods for unified vision-language understanding and generation [29]. Its core insight is that noisy web captions can be improved through caption generation and filtering, producing better supervision for downstream learning. This line of work bridges the gap between static representation learning and text generation.

Flamingo is a major step toward general-purpose multimodal generation [5]. It connects visual inputs to a language model with gated cross-attention and supports interleaved image/video and text inputs. The model's few-shot framing is important because it treats visual tasks as promptable problems rather than fixed-output classifiers. However, Flamingo is not fully open in the sense of public weights and training data, which limits independent reproduction.

### 5.3 Frozen-Backbone Connector Models

Frozen-backbone designs reduce training cost by keeping strong unimodal components fixed. BLIP-2 is central in this category [6]. It uses a Q-Former to query visual features and connect them to a frozen language model. This approach recognizes that high-quality image encoders and language models already contain substantial knowledge; the challenge is learning an efficient bridge between them.

The advantage is efficiency and modularity. Researchers can update the visual encoder, connector, or language model independently. The disadvantage is that connector capacity becomes a bottleneck. If the connector cannot preserve fine-grained visual details, the language model may answer from priors rather than evidence. This is particularly problematic for OCR, counting, small objects, spatial relations, and chart interpretation.

### 5.4 Instruction-Tuned Open MLLMs

Instruction tuning adapts MLLMs to interactive user behavior. LLaVA is a representative open system that combines a vision encoder, projection layer, language model, and visual instruction data [8]. MiniGPT-4 follows a related goal of aligning a visual encoder with a capable language model through a compact training recipe [10]. InstructBLIP extends BLIP-2 toward general-purpose instruction following [9].

The main benefit is usability. Users can ask natural questions about images and receive natural-language answers. The main risk is superficial fluency. A model may learn the style of helpful answers without robust grounding. Instruction tuning therefore requires careful data construction, visual grounding checks, and hallucination evaluation.

### 5.5 Expanded Capability Models

Qwen-VL extends the open MLLM landscape with attention to localization, text reading, and broader multimodal tasks [11]. InternVL studies scale in vision foundation models and alignment for generic vision-language tasks [40]. CogVLM introduces a visual expert design to add visual capability while preserving language ability [7]. These systems show that MLLM architecture is not converging to one simple pattern. Instead, the field is testing several hypotheses about how to best preserve both visual detail and language competence.

Closed frontier systems such as GPT-4V also influence the field [12]. They provide evidence that high-capability multimodal assistants are possible, but their closed nature creates evaluation and reproducibility limits. Researchers can test inputs and outputs, but they cannot inspect training data, architecture, alignment methods, or internal error causes. This makes open systems essential for scientific analysis even when closed systems may perform strongly.

### 5.6 Visual Encoders and Feature Granularity

The visual encoder determines what information is available to the language model before any connector or reasoning step occurs. Many systems reuse pretrained vision transformers or CLIP-like encoders because they provide strong semantic features. This choice is practical, but it can bias the model toward global image semantics. A global representation may capture that an image contains a classroom, laboratory, or road scene, but it may not preserve the exact small text, object count, or spatial relation needed to answer a specific question. For MLLMs, this means encoder quality must be judged not only by classification or retrieval transfer but also by fine-grained evidence preservation.

Feature granularity matters in at least four cases. First, OCR-heavy tasks require the model to preserve character-level or word-level information. If the visual encoder was not optimized for text recognition, the language model may guess plausible text from context. Second, chart and diagram tasks require the model to preserve lines, axes, labels, legends, and relative positions. A semantically strong encoder can still fail if it compresses the image into object categories. Third, counting tasks require object individuation rather than only category recognition. Fourth, spatial reasoning requires relational structure: left/right, above/below, containment, occlusion, and relative size. These demands explain why later systems increasingly emphasize high-resolution inputs, tiling strategies, localization data, and visual grounding.

There is also a tension between input resolution and compute. Increasing image resolution can preserve more information, but it increases the number of visual tokens and the cost of attention. Some systems crop, tile, or select regions; others compress visual features into learned query tokens. Each approach changes what evidence reaches the language model. A reproducible report should therefore specify image resolution, preprocessing, cropping, resizing, patch size, and visual token count when experiments are performed. Without these details, two evaluations of the same model family may not be comparable.

### 5.7 Connectors as Information Bottlenecks

The connector is often the least visible but most important part of an MLLM. It converts visual features into a form that the language model can consume. A simple projection layer is easy to train and deploy, but it may force visual information through a narrow channel. A Q-Former or cross-attention module is more expressive, but it introduces additional parameters, training stages, and design choices. A visual expert module may preserve more visual reasoning capacity, but it can complicate integration with the language backbone.

From an information-flow perspective, the connector must solve three problems. It must align feature distributions so that visual embeddings are meaningful to the language model. It must select or preserve visual details relevant to the instruction. It must do this without damaging the language model's prior abilities. This last point is important because multimodal fine-tuning can cause regressions in text-only tasks if the language model is updated aggressively on narrow visual data. Frozen-backbone approaches reduce that risk, but they rely heavily on connector quality.

Connectors also influence interpretability. If the connector emits a small number of learned query tokens, it can be difficult to determine which image regions influenced the answer. If the model uses cross-attention over many visual tokens, attention maps may provide partial diagnostic information, although attention is not a complete explanation. If the model uses external tools for OCR or detection, the reasoning chain may be more auditable. Future MLLM research should therefore report connector structure with enough detail to support both reproduction and error analysis.

### 5.8 Language Backbones and Multimodal Transfer

The language backbone supplies linguistic fluency, world knowledge, reasoning patterns, and instruction-following behavior. It also supplies biases. When visual evidence is ambiguous, a strong language model may complete the answer from textual priors. This can be useful when a task requires commonsense inference, but it is dangerous when the model must strictly report visible content. For example, a model may infer that a person in a laboratory is wearing protective goggles even if the goggles are not visible. The problem is not only visual weakness; it is the interaction between visual uncertainty and language-model confidence.

Language backbone choice affects multilingual behavior, domain knowledge, safety behavior, context length, and reasoning style. A model trained primarily on English data may underperform on multilingual OCR or non-English visual instructions. A model with strong code or math training may better interpret diagrams and equations, but only if the visual connector provides accurate symbols. A model with stronger safety tuning may refuse some sensitive visual questions, but may also over-refuse benign tasks. These differences mean that architecture comparisons should specify the language model, version, parameter scale, tokenizer, context length, and fine-tuning recipe.

MLLMs also raise the question of whether reasoning should happen entirely inside the language model. For many tasks, external tools may be more reliable. OCR engines, object detectors, mathematical solvers, chart parsers, and retrieval systems can provide structured evidence. An MLLM can then reason over this evidence. This tool-augmented direction may improve reliability, but it changes the system from a single model to a pipeline. Reproducibility then requires documenting every component, not only the neural model.

### 5.9 Open and Closed Model Trade-offs

Open MLLMs support scientific inspection. Researchers can examine architecture, run ablations, test prompts, evaluate datasets, and study failure cases. Open weights also allow domain adaptation and deployment in controlled environments. However, open models may be constrained by compute, data access, and safety resources. Their training data may be less diverse or less carefully filtered than proprietary systems.

Closed MLLMs can benefit from larger training runs, private data pipelines, and extensive safety work. They may set strong capability baselines for public comparison. However, closed systems create uncertainty. If a closed model performs well on a benchmark, researchers cannot easily determine whether the result comes from architecture, data scale, instruction tuning, benchmark contamination, or post-training alignment. System cards and technical reports help, but they do not replace reproducible methods. This survey therefore treats closed systems as important empirical references but not as fully reproducible scientific artifacts.

The open-closed distinction also affects education. For a course report, an open reproducible package is preferable because the instructor and readers can inspect the work. If the paper cites a closed system, it should clearly state what is known and unknown. Claims about closed models should be limited to documented behavior, official reports, or independently executed evaluations.

Table 1. Representative MLLM architecture taxonomy.

| stage | examples | main pattern | role in survey |
| --- | --- | --- | --- |
| Pre-LLM VLP | ViLBERT, LXMERT, UNITER, Oscar, VinVL | cross-modal encoder pretraining | foundation for image-text representation learning |
| Open-vocabulary alignment | CLIP, ALIGN, OFA, PaLI | contrastive or sequence-to-sequence scaling | transferable visual-language representations |
| Frozen-connector MLLMs | Flamingo, BLIP-2, MiniGPT-4 | vision encoder connected to frozen or partly frozen LLM | efficient multimodal generation and few-shot adaptation |
| Instruction-tuned MLLMs | InstructBLIP, LLaVA, Qwen-VL, mPLUG-Owl | visual instruction tuning and alignment | assistant-style multimodal interaction |
| Grounded and visual-expert MLLMs | Kosmos-2, Shikra, Ferret, CogVLM, InternVL | grounding, localization, expert modules, and scale | fine-grained visual evidence and grounding reliability |
| Expanded modality systems | PaLM-E, Video-LLaVA, Visual ChatGPT, GPT-4V | embodiment, video, tool use, or closed frontier alignment | broader deployment and reproducibility trade-offs |

Figure 1. Timeline distribution of representative MLLM-related models.

![Figure 1](figures/model_timeline.png)

## 6. Training Paradigms

### 6.1 Image-Text Contrastive Pretraining

Contrastive pretraining aligns images and text at the representation level. It is scalable, works with large noisy image-caption datasets, and supports retrieval and classification. Its limitation is that it usually captures global correspondence rather than detailed reasoning. For example, matching an image with a caption does not require the model to count objects accurately, read small text, or reason through a diagram. Contrastive pretraining is therefore a strong foundation but not a complete solution.

### 6.2 Captioning and Generative Pretraining

Generative pretraining teaches models to produce text from visual input. Captioning encourages the model to describe salient objects, attributes, and relations. However, captions are often incomplete. A caption may ignore details that are crucial for a later question. If a model is trained primarily on captions, it may underperform on tasks that require reading text, locating small objects, or answering unusual questions. BLIP-style bootstrapping addresses part of this issue by improving caption quality [29].

### 6.3 Connector Alignment

Connector alignment trains a module that maps visual features into the language model's input space. This is computationally attractive because the large language model can remain frozen. BLIP-2 is an important example [6]. The key question is what information the connector preserves. A small projection may be efficient but lossy. A query transformer can learn task-relevant summaries but may miss details outside its learned query capacity. Cross-attention can be expressive but computationally heavier.

### 6.4 Visual Instruction Tuning

Visual instruction tuning uses multimodal conversations or question-answer pairs. LLaVA demonstrated the effectiveness of this approach for creating open visual assistants [8]. InstructBLIP refined instruction tuning in a BLIP-style architecture [9]. The training data may include human-written examples, synthetic examples generated by stronger models, or mixtures of task datasets reformatted as instructions.

The strength of instruction tuning is behavioral alignment. It teaches the model to respond in a way users expect. The weakness is dependency on data quality. If training examples contain ungrounded claims, models can learn to hallucinate. If examples overrepresent certain domains, models may fail in underrepresented visual styles. If synthetic data is generated by a closed model, errors and biases may transfer into the open model.

### 6.5 Safety and Refusal Alignment

MLLMs introduce safety problems beyond text-only systems. A visual model may answer questions about people, medical images, identity, documents, or dangerous physical procedures. Safety alignment must consider both the text prompt and the visual content. GPT-4V's system card emphasizes safety evaluation and deployment considerations for visual inputs [12]. Open research still needs stronger public benchmarks for visual privacy, sensitive inference, medical uncertainty, and malicious multimodal prompting.

## 7. Benchmark Evaluation and Comparative Analysis

### 7.1 Classical VQA Benchmarks

VQA introduced open-ended question answering over images [13]. It remains historically important because it framed visual understanding as language-conditioned prediction. However, early VQA datasets can contain language biases, making it possible for models to answer some questions from text priors alone. GQA was designed to emphasize compositional reasoning and scene-graph structure [14]. OK-VQA requires external knowledge beyond visible image content [15]. Together, these benchmarks show that visual question answering is not a single skill: it may require perception, language priors, compositional reasoning, and world knowledge.

### 7.2 Educational and Scientific Reasoning

ScienceQA evaluates multimodal science-question reasoning and explanations [16]. It is relevant because scientific tasks often combine text, diagrams, symbolic notation, and commonsense knowledge. MLLMs are attractive for education because they can potentially explain visual material, but this also raises risks. An incorrect explanation can be more harmful than a simple wrong label because it may appear authoritative. Therefore, education-oriented benchmarks should evaluate not only answer correctness but also reasoning quality and uncertainty.

### 7.3 Broad MLLM Diagnostic Benchmarks

MME provides a broad diagnostic benchmark for perception and cognition in MLLMs [17]. MMBench asks whether multimodal models are all-around players and emphasizes systematic evaluation [18]. MM-Vet evaluates integrated capabilities, recognizing that real tasks often require multiple skills at once [19]. These benchmarks are useful because they move beyond narrow datasets. However, they should not be treated as complete measures of intelligence. Benchmark construction choices, answer formats, prompt templates, and model-specific preprocessing can influence outcomes.

### 7.4 Expert-Domain and Mathematical Reasoning

MMMU evaluates expert-level multimodal understanding across disciplines [20]. MathVista focuses on mathematical reasoning in visual contexts [22]. These benchmarks are important because many practical applications require more than object recognition. A model must interpret charts, diagrams, geometric relations, equations, and domain-specific terminology. Such tasks stress both visual parsing and symbolic reasoning. They also reveal the limitations of language priors: a fluent explanation is not enough if the model misreads a graph or diagram.

### 7.5 Hallucination Evaluation

Hallucination is a central MLLM failure mode. POPE evaluates object hallucination by probing whether models assert the presence of objects not in the image [21]. This problem is structurally connected to language-model priors. If a model sees a kitchen, it may mention common kitchen objects even when they are absent. If it sees a street, it may infer traffic signs or vehicles from context. Such behavior is dangerous in settings where visual evidence matters, including medicine, inspection, navigation, and legal documents.

Hallucination evaluation should separate several cases: nonexistent object mentions, incorrect attributes, wrong spatial relations, false OCR, unsupported causal explanations, and overconfident domain claims. A benchmark that only checks object presence cannot capture all hallucination types. Future evaluation should also test whether models can say "not visible", "uncertain", or "the image does not provide enough evidence" when appropriate.

Table 2. Representative MLLM benchmark taxonomy.

| benchmark group | examples | primary capability | evaluation risk |
| --- | --- | --- | --- |
| General VQA | VQA, GQA, OK-VQA | image-conditioned question answering | language priors and dataset bias |
| OCR and chart reasoning | TextVQA, ChartQA | reading scene text and structured graphics | small text, chart parsing, and answer normalization |
| Science and mathematics | ScienceQA, MathVista | diagram-grounded and visual mathematical reasoning | fluent but unsupported reasoning |
| Broad diagnostics | MME, MMBench, MM-Vet | perception, cognition, and integrated skills | prompt sensitivity and scoring differences |
| Hallucination checks | POPE | detecting unsupported object claims | coverage beyond object presence remains limited |
| Expert-domain reasoning | MMMU | college-level multidisciplinary multimodal reasoning | contamination and domain coverage |

Figure 2. Timeline distribution of representative MLLM benchmarks.

![Figure 2](figures/benchmark_timeline.png)

### 7.6 Comparative Analysis Without Fabricated Scores

This survey intentionally does not provide a leaderboard. A valid leaderboard requires controlled execution, exact model versions, prompts, decoding parameters, datasets, and scoring scripts. Without these details, benchmark numbers can be misleading. Even numbers copied from source papers must be interpreted carefully because models may have different training data, evaluation preprocessing, and prompt formats.

Qualitatively, the literature supports several defensible comparisons. Contrastive models such as CLIP are strong representation learners but are not full conversational MLLMs [3]. Flamingo and BLIP-2 demonstrate that frozen components and learned connectors can produce strong multimodal generation and few-shot adaptation [5] [6]. LLaVA and InstructBLIP show that instruction tuning is essential for assistant-like behavior [8] [9]. Qwen-VL, InternVL, and CogVLM represent later efforts to expand grounding, scale, and visual expertise [11] [40] [7]. Closed systems such as GPT-4V remain important external references but cannot be fully reproduced from public information [12].

### 7.7 Benchmark Design Risks

Benchmark design strongly shapes research conclusions. A benchmark that uses multiple-choice questions may be easy to score, but it can reward test-taking strategies. A model may eliminate unlikely options using language priors rather than interpreting the image. Conversely, an open-ended benchmark may better reflect real user interaction, but automatic scoring becomes difficult because many correct answers may be phrased differently. Human evaluation can help, but it is expensive and may introduce annotator disagreement.

Prompt sensitivity is another major risk. An MLLM may answer differently when asked "What is in the image?", "Describe the image carefully", or "Answer with one word." Some benchmarks provide fixed prompts, while others leave prompt choice to evaluators. If prompt templates differ across papers, results cannot be compared directly. Decoding settings also matter. Greedy decoding, beam search, temperature sampling, and maximum-token limits can change outputs. Therefore, a reproducible benchmark report should specify prompt templates, answer extraction rules, decoding parameters, image preprocessing, model checkpoint, and scoring code.

Dataset contamination is especially difficult for MLLMs. Large-scale web training may include benchmark images, questions, answers, or near duplicates. Even if exact benchmark data is absent, related examples may be present. Contamination can inflate performance and make it unclear whether the model learned general reasoning or memorized patterns. Stronger reporting should include contamination checks where possible, such as near-duplicate image search, text overlap analysis, and evaluation on newly collected private or time-separated data.

Another risk is benchmark narrowness. A model can score well on common object recognition but fail at medical images, satellite imagery, engineering diagrams, or local cultural contexts. A model can answer English visual questions but fail multilingual instructions. A model can perform well on static images but fail video sequences requiring temporal reasoning. For this reason, benchmark results should be interpreted as evidence about the tested distribution, not universal ability.

### 7.8 Error Taxonomy for MLLMs

A useful evaluation framework should classify errors, not only count them. In MLLMs, errors often arise from different sources. Perception errors occur when the model fails to detect visible objects, text, attributes, or spatial relations. Grounding errors occur when the model identifies a correct concept but attaches it to the wrong region or entity. Reasoning errors occur when visual evidence is parsed correctly but the conclusion is wrong. Knowledge errors occur when the task requires external facts and the language model supplies incorrect knowledge. Instruction-following errors occur when the answer does not match the requested format or scope. Safety errors occur when the model provides harmful, privacy-invasive, or overconfident responses.

This taxonomy matters because different interventions address different errors. A better visual encoder may reduce perception errors but not reasoning errors. More instruction tuning may improve format compliance but increase hallucination if the data is weakly grounded. External OCR tools may improve text reading but not general spatial reasoning. Safety tuning may reduce harmful answers but not improve visual accuracy. Without error categorization, researchers may select the wrong solution for the observed failure.

The most dangerous errors are often plausible. If a model gives an obviously nonsensical answer, users may distrust it. If it gives a fluent, detailed, but unsupported explanation, users may accept it. This is why hallucination and calibration should be evaluated together. A reliable MLLM should know when visual evidence is insufficient and should express uncertainty in a way that users can understand.

### 7.9 Reproducible Evaluation Protocol

A controlled future experiment for this project could follow a transparent protocol. First, select open models with fixed released checkpoints. Second, record exact model identifiers, commit hashes, dependencies, and licenses. Third, download benchmarks from official sources and store checksums or version identifiers. Fourth, use a fixed image preprocessing pipeline. Fifth, define prompts in a configuration file rather than hard-coding them. Sixth, run inference with fixed decoding settings. Seventh, save raw model outputs before scoring. Eighth, score with released scripts or clearly documented local scripts. Ninth, report both aggregate metrics and representative error cases. Tenth, publish the full logs and configuration files.

Such a protocol would allow the paper to include quantitative results in a future version. Until then, the correct scientific choice is to avoid benchmark numbers. The current paper therefore provides taxonomy and analysis only.

## 8. Reproducibility Framework

### 8.1 What Reproducibility Means for a Survey

Reproducibility in an experimental MLLM paper means that another researcher can rerun training or evaluation and obtain the same or meaningfully comparable results. Reproducibility in a survey is different but still important. A survey should allow readers to inspect which papers were included, how categories were assigned, how tables were generated, and where claims originate. This local package follows that principle by separating manuscript text, bibliography, structured taxonomy tables, and scripts.

The bibliography file is a single source of truth for citation metadata. The paper cites entries using stable keys. The validation script checks that every cited key exists in the bibliography. This does not prove that every citation is perfectly formatted, but it prevents a common reproducibility problem: citing a source that is absent from the reference list. The model and benchmark taxonomy tables are stored as CSV files, making them easier to update and diff than manually formatted tables.

### 8.2 What Would Be Required for Original Experiments

If this project is extended from survey to original empirical research, the reproducibility requirements become much stricter. The repository would need dataset preparation scripts, exact download instructions, preprocessing code, training code, evaluation code, inference code, configuration files, model checkpoint identifiers, hardware descriptions, software versions, and random seeds. It would also need raw logs and result files. The paper would need to state optimizer, learning rate, batch size, number of epochs, weight decay, scheduler, precision mode, GPU type, memory, and total training time.

For MLLM evaluation, additional fields are needed: image resolution, crop or tiling method, prompt template, system prompt, decoding temperature, top-p, maximum output tokens, answer normalization, scoring rule, and benchmark version. If API-based closed models are used, the paper should record model name, API version, access date, and any system instructions. Because API models can change over time, outputs should be saved locally as evidence.

### 8.3 Data and License Responsibility

MLLM research often uses datasets with different licenses and privacy expectations. A reproducible repository should not automatically redistribute datasets unless redistribution is permitted. Instead, it should provide official links, citation information, checksums when possible, and scripts that place downloaded files into expected directories. This is why the current `datasets/` folder contains only notes and empty subdirectories. Adding data without license review would weaken the project, not strengthen it.

Synthetic instruction data also requires documentation. If prompts or answers are generated by another model, the generating model, prompt, filtering process, and date should be recorded. Synthetic data can be useful, but it can also amplify errors or create circular evaluation if the same model family is later used as a judge. A rigorous project should separate human annotations, synthetic annotations, and benchmark labels.

### 8.4 Public Repository Status

The template names `https://github.com/IssaIssa-tech/AI-final-report` as the intended repository. The URL exists, but it must contain the actual files before the paper claims public availability. This distinction is not administrative; it is part of research integrity. A reader should be able to open the link, inspect the code, and run the scripts. Until upload is complete, the paper should say that the local package is prepared and the public repository is pending population.

## 9. Ethical, Safety, and Deployment Considerations

### 9.1 Visual Privacy and Sensitive Inference

MLLMs process images that may contain faces, homes, identity documents, license plates, medical information, location clues, workplace materials, or personal messages. This makes privacy risk more direct than in text-only systems. A text prompt may describe a person, but an image can reveal appearance, environment, socioeconomic signals, health conditions, and private context without the person's explicit consent. A deployed MLLM should therefore avoid unnecessary sensitive inference. It should not infer identity, protected attributes, medical diagnoses, or private facts from images unless the application has a justified, consent-based, and legally appropriate basis.

The privacy problem is not limited to final answers. Uploaded images may be stored in logs, used for monitoring, or retained for future training. A reproducible research paper should disclose whether visual inputs are stored, whether they are shared with third-party APIs, and whether they are used for training. In a course project, the safest default is to use public benchmark data according to its license and avoid uploading private images to external services. If a closed API is used, the paper should document the service terms and data-retention settings available at the time of evaluation.

### 9.2 Reliability in High-Stakes Domains

MLLMs are attractive for domains such as medicine, education, law, engineering, and public safety because they can explain visual material in natural language. The same property makes errors more consequential. A medical-image explanation, a legal-document interpretation, or a safety-inspection judgment can influence user decisions. Current MLLMs are not reliable enough to be treated as autonomous experts in these settings without validation, oversight, and domain-specific controls.

High-stakes deployment should require conservative behavior. The model should distinguish visible evidence from inference, state uncertainty, and recommend professional review when appropriate. It should avoid presenting guesses as facts. For example, if an image is low resolution, partially occluded, or outside the model's training distribution, the system should say so. Benchmark accuracy alone is insufficient because rare failures can be severe. Evaluation should include stress tests, adversarial examples, out-of-distribution inputs, and human review of failure cases.

### 9.3 Hallucination and User Trust

Hallucination is not only a technical error; it is a trust problem. Users often assume that a detailed answer reflects detailed perception. In MLLMs, this assumption can be false. A model may provide a confident description because the language model knows what usually appears in similar scenes, not because the visual encoder captured the evidence. This risk is especially serious when outputs include numbers, text transcription, identity claims, scientific explanations, or procedural instructions.

User interfaces can reduce or amplify hallucination harm. A system that shows uncertainty, highlights image regions, or cites extracted OCR evidence gives users more opportunity to verify. A system that presents a polished paragraph with no evidence may invite overtrust. Therefore, model design and interface design should be considered together. Future MLLM systems should expose confidence, evidence, and limitations where possible, especially for tasks requiring exact visual grounding.

### 9.4 Bias and Representation

MLLMs inherit biases from both visual and textual data. Visual datasets may underrepresent certain regions, occupations, skin tones, clothing styles, scripts, cultural artifacts, and living environments. Language data may encode stereotypes and social assumptions. When these modalities interact, bias can appear in subtle ways. A model may describe people differently based on appearance, infer occupations from gendered cues, or fail to read non-Latin scripts. It may also perform better on images from regions and cultures that dominate training data.

Bias evaluation should therefore include diverse visual contexts and languages. It should not only measure whether the model recognizes objects but also how it describes people and social situations. In educational settings, biased explanations can disadvantage students whose materials, languages, or cultural references are underrepresented. In public deployment, biased visual inference can cause reputational or material harm. A responsible survey should acknowledge these risks even when it does not perform new bias experiments.

### 9.5 Environmental and Compute Costs

Training large MLLMs can require substantial compute, energy, and engineering resources. Even fine-tuning and evaluation can be expensive when models process high-resolution images or long visual-token sequences. This creates an access gap: organizations with large compute budgets can train frontier models, while universities and smaller labs may rely on open checkpoints or APIs. Efficient connectors, frozen-backbone methods, parameter-efficient tuning, and careful evaluation design are therefore not only engineering conveniences; they are also important for broader research participation.

For a course project, it is appropriate to avoid unnecessary training when the research question is survey-based. Running large experiments without a clear hypothesis would waste resources and may not improve the paper. The present project uses lightweight scripts because the goal is literature organization and reproducibility, not model optimization.

### 9.6 Practical Deployment Guidance

Practical MLLM deployment should follow a layered approach. First, define the task and risk level. A casual image-description assistant has different requirements from a medical triage tool. Second, choose models and tools appropriate to the task. If exact OCR is required, a specialized OCR component may be necessary. If mathematical diagram reasoning is required, a symbolic or programmatic solver may be useful. Third, evaluate on domain-specific data, not only general benchmarks. Fourth, monitor failures and update prompts, tools, or models based on evidence. Fifth, provide user-facing uncertainty and escalation paths.

This guidance reinforces the main argument of the survey: MLLMs should not be evaluated only as general chatbots. They are multimodal systems whose reliability depends on perception, grounding, language reasoning, safety alignment, interface design, and deployment context.

## 10. Discussion

The MLLM field is shaped by a trade-off between capability, openness, and reliability. Closed frontier systems can deliver strong user-facing behavior, but they offer limited scientific transparency. Open models make analysis possible, but they may lag closed models in training scale, data quality, and safety alignment. Surveys and benchmarks must therefore avoid treating capability as a single number. A model's usefulness depends on what it sees, how it reasons, whether it admits uncertainty, and whether its training and evaluation can be audited.

Architecture choices reveal another trade-off. Freezing large backbones reduces compute and makes modular development possible. However, it may limit deep cross-modal integration. End-to-end training can improve integration but is expensive and difficult to reproduce. Query transformers, projection layers, cross-attention, and visual experts are all attempts to balance this trade-off. No design has solved all challenges. Fine-grained OCR, spatial reasoning, long video understanding, chart analysis, and multimodal mathematical reasoning remain difficult.

Training data quality is likely as important as architecture. Web-scale image-text pairs provide breadth, but they are noisy and often weakly grounded. Instruction data improves interaction, but synthetic data can import errors. Benchmark data encourages measurable progress, but repeated leaderboard optimization can reduce generalization. A robust MLLM training pipeline should therefore combine diverse data, grounding checks, adversarial evaluation, and explicit uncertainty behavior.

Evaluation remains the most fragile part of MLLM research. Many benchmarks test a mixture of skills, making it hard to diagnose failure. Multiple-choice benchmarks are easier to score but can be sensitive to answer priors. Open-ended benchmarks better match real use but require reliable automatic or human evaluation. Hallucination benchmarks are essential but must cover more than object presence. Expert-domain benchmarks are valuable but risk contamination if tasks or solutions appear in training data. A mature evaluation ecosystem should include standard prompts, released scoring scripts, contamination checks, and error taxonomies.

For ChinaXiv-style reproducibility, the key requirement is transparency. A survey paper should state exactly what it does and does not do. This paper does not execute model benchmarks. It provides structured literature organization and reproducibility scripts for taxonomy tables. If future work adds experiments, it must include dataset access instructions, preprocessing code, model versions, decoding parameters, hardware, software versions, random seeds, logs, and raw outputs.

## 11. Limitations

This paper has several limitations. First, it is a survey and does not contain new experiments. Therefore, it cannot make empirical claims about comparative model performance beyond what is supported by cited literature. Second, the field changes rapidly, and new MLLMs may appear after this draft. The bibliography should be updated before final submission. Third, the paper focuses primarily on image-language models and does not deeply cover audio, speech, 3D perception, robotics, or long-form video. Fourth, closed systems such as GPT-4V cannot be fully analyzed because training details and model weights are not public [12]. Fifth, the taxonomy simplifies complex systems into categories; some models fit multiple categories at once.

There are also reproducibility limitations. The local package contains scripts for citation validation and table generation, but it does not include benchmark datasets, model checkpoints, or training logs. This is appropriate for a survey but insufficient for an original experimental paper. The public GitHub repository URL exists, but the repository must be populated before the paper claims that all materials are publicly available.

## 12. Conclusion

Multimodal large language models represent a major shift in artificial intelligence research. They combine visual perception, language generation, instruction following, and reasoning in a single interface. Their development builds on transformers, large language models, contrastive vision-language learning, generative pretraining, connector-based alignment, and visual instruction tuning. Representative systems such as CLIP, Flamingo, BLIP-2, InstructBLIP, LLaVA, MiniGPT-4, Qwen-VL, InternVL, CogVLM, and GPT-4V illustrate the diversity of architecture and alignment strategies.

The central lesson is that MLLM progress cannot be understood only through benchmark scores. Architecture, training data, grounding, hallucination behavior, safety, openness, and reproducibility all matter. Benchmarks such as VQA, GQA, OK-VQA, ScienceQA, MME, MMBench, MM-Vet, MMMU, POPE, and MathVista evaluate important capabilities, but each covers only part of the problem. Reliable deployment requires broader evaluation and more transparent reporting.

This survey provides a reproducible foundation for a ChinaXiv-style final report. It includes a verified bibliography, taxonomy tables, scripts, and explicit limitations. The work can be extended by adding updated literature, deeper benchmark analysis, or executed experiments with full reproducibility artifacts.

## 13. Future Work

Future work should extend this survey in five directions. First, the bibliography should be updated with newly peer-reviewed MLLM papers and benchmark studies. Second, the taxonomy should be expanded to cover video-language, audio-language, document intelligence, robotics, and multimodal agents. Third, a controlled benchmark study could be added using open models, fixed prompts, exact decoding parameters, released scoring scripts, and raw output logs. Fourth, hallucination analysis should be broadened from object presence to OCR errors, spatial relation errors, causal overclaiming, and uncertainty calibration. Fifth, the public GitHub repository should be populated with the full local package so that the Code Availability section becomes externally verifiable.

## Code Availability

The intended public repository for this work is:

https://github.com/IssaIssa-tech/AI-final-report

At the time this local package was prepared, the GitHub repository existed but was empty. Therefore, this paper should not claim that the public repository already contains the reproducibility materials until the local files are uploaded. The local package contains:

- `paper/paper.md`: survey manuscript
- `paper/references.bib`: bibliography
- `paper/tables/model_taxonomy.csv`: model taxonomy
- `paper/tables/benchmark_taxonomy.csv`: benchmark taxonomy
- `scripts/generate_tables.py`: table generation script
- `scripts/validate_references.py`: citation-key validation script
- `requirements.txt` and `environment.yml`: software environment files

## Final Verification Checklist

- Required sections are present: title, authors, abstract, keywords, introduction, related work, theoretical foundation, methodology, comparative analysis, discussion, limitations, conclusion, future work, references, and code availability.
- References are real works from peer-reviewed venues, arXiv preprints, or official technical reports.
- No new benchmark numbers are reported.
- No experiments are claimed.
- Implementation details for the included scripts are provided.
- The public GitHub repository must be populated before public code availability is claimed.

## References

[1] Vaswani Ashish, Shazeer Noam, Parmar Niki, et al. Attention Is All You Need[C]. Advances in Neural Information Processing Systems, 2017.

[2] Brown Tom B., Mann Benjamin, Ryder Nick, et al. Language Models are Few-Shot Learners[J]. Advances in Neural Information Processing Systems, 2020.

[3] Radford Alec, Kim Jong Wook, Hallacy Chris, et al. Learning Transferable Visual Models From Natural Language Supervision[C]. International Conference on Machine Learning, 2021.

[4] Jia Chao, Yang Yinfei, Xia Ye, et al. Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision[C]. International Conference on Machine Learning, 2021.

[5] Alayrac Jean-Baptiste, Donahue Jeff, Luc Pauline, et al. Flamingo: A Visual Language Model for Few-Shot Learning[C]. Advances in Neural Information Processing Systems, 2022.

[6] Li Junnan, Li Dongxu, Savarese Silvio, et al. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models[C]. International Conference on Machine Learning, 2023.

[7] Wang Weihan, Lv Qingsong, Yu Wenmeng, et al. CogVLM: Visual Expert for Pretrained Language Models[C]. Advances in Neural Information Processing Systems, 2024.

[8] Liu Haotian, Li Chunyuan, Wu Qingyang, et al. Visual Instruction Tuning[C]. Advances in Neural Information Processing Systems, 2023.

[9] Dai Wenliang, Li Junnan, Li Dongxu, et al. InstructBLIP: Towards General-Purpose Vision-Language Models with Instruction Tuning[C]. Advances in Neural Information Processing Systems, 2023.

[10] Zhu Deyao, Chen Jun, Shen Xiaoqian, et al. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models[C]. International Conference on Learning Representations, 2024.

[11] Bai Jinze, Bai Shuai, Yang Shusheng, et al. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond[J]. arXiv preprint arXiv:2308.12966, 2023.

[12] OpenAI. GPT-4V(ision) System Card[R]. OpenAI, 2023. https://openai.com/research/gpt-4v-system-card

[13] Antol Stanislaw, Agrawal Aishwarya, Lu Jiasen, et al. VQA: Visual Question Answering[C]. IEEE International Conference on Computer Vision, 2015.

[14] Hudson Drew A., Manning Christopher D. GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[15] Marino Kenneth, Rastegari Mohammad, Farhadi Ali, et al. OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[16] Lu Pan, Mishra Swaroop, Xia Tanglin, et al. Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering[C]. Advances in Neural Information Processing Systems, 2022.

[17] Fu Chaoyou, Chen Peixian, Shen Yunhang, et al. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models[J]. arXiv preprint arXiv:2306.13394, 2023.

[18] Liu Yuan, Duan Haodong, Zhang Yuanhan, et al. MMBench: Is Your Multi-modal Model an All-around Player?[J]. arXiv preprint arXiv:2307.06281, 2023.

[19] Yu Weihao, Yang Zhengyuan, Li Linjie, et al. MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities[J]. arXiv preprint arXiv:2308.02490, 2023.

[20] Yue Xiang, Ni Yuansheng, Zhang Kai, et al. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI[J]. arXiv preprint arXiv:2311.16502, 2023.

[21] Li Yifan, Du Yifan, Zhou Kun, et al. Evaluating Object Hallucination in Large Vision-Language Models[J]. arXiv preprint arXiv:2305.10355, 2023.

[22] Lu Pan, Bansal Hritik, Xia Tony, et al. MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts[J]. arXiv preprint arXiv:2310.02255, 2023.

[23] Lu Jiasen, Batra Dhruv, Parikh Devi, et al. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks[C]. Advances in Neural Information Processing Systems, 2019.

[24] Tan Hao, Bansal Mohit. LXMERT: Learning Cross-Modality Encoder Representations from Transformers[C]. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 2019.

[25] Chen Yen-Chun, Li Linjie, Yu Licheng, et al. UNITER: UNiversal Image-TExt Representation Learning[C]. European Conference on Computer Vision, 2020.

[26] Li Xiujun, Yin Xi, Li Chunyuan, et al. Oscar: Object-Semantics Aligned Pre-training for Vision-Language Tasks[C]. European Conference on Computer Vision, 2020.

[27] Zhang Pengchuan, Li Xiujun, Hu Xiaowei, et al. VinVL: Revisiting Visual Representations in Vision-Language Models[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021.

[28] Wang Peng, Yang An, Men Rui, et al. OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework[C]. International Conference on Machine Learning, 2022.

[29] Li Junnan, Li Dongxu, Xiong Caiming, et al. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation[C]. International Conference on Machine Learning, 2022.

[30] Chen Xi, Wang Xiao, Changpinyo Soravit, et al. PaLI: A Jointly-Scaled Multilingual Language-Image Model[J]. arXiv preprint arXiv:2209.06794, 2022.

[31] Driess Danny, Xia Fei, Sajjadi Mehdi S. M., et al. PaLM-E: An Embodied Multimodal Language Model[C]. International Conference on Machine Learning, 2023.

[32] Huang Shaohan, Dong Li, Wang Wenhui, et al. Language Is Not All You Need: Aligning Perception with Language Models[J]. arXiv preprint arXiv:2302.14045, 2023.

[33] Peng Zhiliang, Wang Wenhui, Dong Li, et al. Kosmos-2: Grounding Multimodal Large Language Models to the World[J]. arXiv preprint arXiv:2306.14824, 2023.

[34] Ye Qinghao, Xu Haiyang, Xu Guohai, et al. mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality[J]. arXiv preprint arXiv:2304.14178, 2023.

[35] Chen Keqin, Zhang Zhao, Zeng Weili, et al. Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic[J]. arXiv preprint arXiv:2306.15195, 2023.

[36] You Haoxuan, Zhang Haotian, Gan Zhe, et al. Ferret: Refer and Ground Anything Anywhere at Any Granularity[C]. International Conference on Learning Representations, 2024.

[37] Lin Ji, Yin Hongxu, Ping Wei, et al. VILA: On Pre-training for Visual Language Models[J]. arXiv preprint arXiv:2312.07533, 2023.

[38] Lin Bin, Ye Yang, Zhu Bin, et al. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection[C]. Conference on Empirical Methods in Natural Language Processing, 2024.

[39] Wu Chenfei, Yin Shengming, Qi Weizhen, et al. Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models[J]. arXiv preprint arXiv:2303.04671, 2023.

[40] Chen Zhe, Wu Jiannan, Wang Wenhai, et al. InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

[41] Singh Amanpreet, Natarajan Vivek, Shah Meet, et al. Towards VQA Models That Can Read[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

[42] Masry Ahmed, Long Do Xuan, Tan Jia Qing, et al. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning[C]. Findings of the Association for Computational Linguistics: ACL 2022, 2022.
