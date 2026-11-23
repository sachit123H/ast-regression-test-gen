# CS.91000 GRADUATION RESEARCH

**Sachit Varshney**

**Under Guidance of:**  
Prof. Dr. Shin Yoo

Korea Advanced Institute of Science and Technology  
05 August 2026

*A graduation research proposal submitted for the requirements of the degree of Bachelor's in Computer Science*

---

> **Post-August 2026 Terminology Revision & Task Clarification Note:**  
> Following critical guidance from advisor Prof. Dr. Shin Yoo on August 7, 2026, the problem formulation of this research was formally updated from generic "regression test generation" to **post-fix bug-reproducing test (BRT) generation from a bug-fixing commit**. In classical testing literature (Yoo & Harman 2012), regression testing denotes re-executing pre-existing suites to ensure unmodified features remain intact. In contrast, our task addresses synthesizing a *new* test case that fails on the buggy (pre-fix) revision and passes on the fixed (post-fix) revision. Once validated by the differential execution oracle, the test is permanently admitted into the suite as a regression guard. This framing is motivated by three realistic software engineering use cases: (1) time-pressured production hotfixes deployed without tests, (2) external open-source contributor PRs lacking tests due to framework unfamiliarity, and (3) edge-case expansion beyond minimal symptom checks. The repository directory name `ast-regression-test-gen` is maintained for git remote commit tracking and path stability.

---

## Contents

- [Abstract](#abstract)
- [1 Motivation](#1-motivation)
- [2 Research Questions](#2-research-questions)
- [3 Background and Related Work](#3-background-and-related-work)
  - [3.1 Regression Test Generation Context](#31-regression-test-generation-context)
  - [3.2 LLM and Agentic Test Generation](#32-llm-and-agentic-test-generation)
  - [3.3 Related Work](#33-related-work)
- [4 Methodology](#4-methodology)
  - [4.1 Dataset](#41-dataset)
  - [4.2 Project Structure and Agentic Loop](#42-project-structure-and-agentic-loop)
  - [4.3 Role of MCP (Model Context Protocol)](#43-role-of-mcp-model-context-protocol)
  - [4.4 GumTree and AST Edit Scripts](#44-gumtree-and-ast-edit-scripts)
  - [4.5 Differential Execution Oracle](#45-differential-execution-oracle)
  - [4.6 Evaluation Metrics](#46-evaluation-metrics)
- [5 Roadmap and Timeline](#5-roadmap-and-timeline)
  - [5.1 Anticipated Risks and Fixes](#51-anticipated-risks-and-fixes)
- [6 Conclusion](#6-conclusion)
- [Bibliography](#bibliography)

---

## Abstract

An agentic test generation system supplies a large language model with some representation of the code under test, and the quality of that representation bounds what the agent can infer. When the task is to generate a regression test for a bug-fixing commit, the change itself must be represented somehow. Two such representations are textual diff (`git diff`) and an Abstract Syntax Tree (AST) edit script. This work asks whether that choice affects the quality of the regression tests an LLM agent generates for Java programs. We construct two conditions that differ only in the change representation, a raw-diff baseline and a normalized AST [1] edit script (generated through GumTree), and hold the agent, the tools, and the feedback identical across both. We implement a Model Context Protocol (MCP) server that lets an autonomous agent iteratively generate, execute, and refine tests against the pre-fix version, and we validate the resulting tests with a differential execution oracle on Defects4J [2]. The two conditions are compared using Fault Diagnosability Gain (FDG) [3].

---

## 1 Motivation

The effectiveness of an LLM on a code task depends on what is placed in its context window. AutoCodeRover [4] navigates a project through its Abstract Syntax Tree rather than as a collection of files, and TestWeaver [5] augments a test generation loop with program slices and execution traces; both report that the representation supplied to the model changes what it produces. A bug-fixing commit can likewise be represented in more than one way. The output of `git diff` is a textual, line-oriented record of what changed, whereas an AST edit script from a tree differencing tool such as GumTree [1] reports typed operations over syntax tree nodes, making node types and method boundaries explicit. We hypothesise that this structural framing helps an agent identify what behaviour the fix changed, and therefore what a regression test should assert. We do not claim that existing agents are limited by their use of raw diffs. We use the raw diff as a constructed baseline because it is available directly from version control and is the minimal textual counterpart to the AST edit script, so the two conditions differ in structure rather than in content.

---

## 2 Research Questions

This research is guided by the following core questions:

- **RQ1.** If an LLM-based test generation agent is provided with structural AST level edit scripts (generated using GumTree) instead of the raw source diffs will it improve its ability to generate regression tests that satisfies the differential execution oracle [Sec. 4.5] for real bug fixing commits?
- **RQ2.** To what extent does AST-level change context improve the Fault Diagnosability Gain (FDG) [Sec. 4.6] of the generated test suite compared to the raw-diff baseline?

---

## 3 Background and Related Work

### 3.1 Regression Test Generation Context

A regression test verifies that a bug fix does not break the existing functionality of the code by re-running those functional and non-functional tests that are directly relevant for the selected section of code that is modified via the bug fix.

### 3.2 LLM and Agentic Test Generation

Recent work uses LLMs to generate unit tests, with modern agentic approaches iteratively generating, executing, and refining tests using execution feedback. Their effectiveness largely depends on the quality of the change context provided.

To evaluate whether that context is well spent, this proposal adopts the Fault Diagnosability Gain (FDG) metric introduced by An and Yoo [3], which quantifies the improvement in the ranking of the faulty statement once a test is added to the suite. FDG was originally proposed to assess test suites produced by search-based generation (EvoSuite); here it is repurposed as the evaluation criterion for comparing raw-diff and AST-edit-script contexts in an LLM agent, rather than as a technique this work is positioned against.

### 3.3 Related Work

Previous studies have explored various dimensions of LLM testing but have overlooked the impact of AST structural data in an agentic workflow. Table 1 summarizes the current landscape and the specific gap this proposal addresses.

| Work | Approach | Relevant work |
| :--- | :--- | :--- |
| **CodaMosa** [6] | Search-based test generation that queries an LLM with the module source when coverage stalls, then deserializes the result back into the search | Demonstrates that LLM-supplied context changes what a generator can reach, but the context is the module under test and the target is coverage, not a fix |
| **TestWeaver** [5] | Iterative LLM test generation enriched with backward slices, a retrieved closest test case, and inline execution annotations | Closest in method: it varies the enrichment supplied to a feedback loop and ablates each component. The enrichment is execution and slice data toward coverage, not a structural view of a change |
| **AutoCodeRover** [4] | LLM agent that retrieves context through AST-based code search APIs, optionally augmented with spectrum-based fault localisation, and emits a patch | Shares the premise that a structural program representation serves an LLM better than file-level text, but tests it on patch generation from an issue report rather than on test generation from a change |
| **LIBRO** [7, 8] | LLM generates bug-reproducing tests from bug reports; candidates filtered by failure on the buggy version, then clustered and ranked by agreement | Input is a natural-language bug report, not structural change context; single-shot generation with ranking rather than iterative execution-feedback repair. This work adopts LIBRO's differential validity criterion. |

*Table 1: Summary of related work and addressed limitations.*

These systems differ mainly in what they place in the model's context window, and several of them report that this choice measurably changes what the model produces. That variable is what this proposal isolates. Existing LLM-based test generation supplies either method-level context or natural-language symptom descriptions. The representation of the bug-fixing change itself, textual versus structural, has not been isolated and compared.

---

## 4 Methodology

### 4.1 Dataset

We draw target commits from Defects4J [2], which provides reproducible builds and a uniform checkout, compile, and test interface across its bug set, so that a pipeline failure is attributable to our implementation rather than to the benchmark. Each case includes the pre-fix and post-fix versions, the raw patch, and the corresponding AST edit script.

Bugs are selected by three filters:
- Both versions build and their suites run
- The fix is localized, meaning its edits fall within a single method
- GumTree parses both versions successfully.

The resulting bug set is identical for both conditions. A bug excluded by any filter is excluded from both, and the number excluded by each filter is reported.

### 4.2 Project Structure and Agentic Loop

The system operates as a closed agentic loop rather than a static prompt generator. The core architecture follows a robust feedback cycle:

1. The agent is initialized with either the raw diff (Control) or the normalized AST edit script (Treatment).
2. The agent proposes a candidate regression test targeting the specific change.
3. The candidate is compiled and executed against the pre-fix version.
4. Feedback is routed back to the agent, based on three execution-derived signals:
   - Compiler output, in full and verbatim.
   - Pre-fix execution outcome: pass or fail. And on failure the failure type and relevant details like (if applicable) exception type, message, and stack trace.
   - Coverage, measured on the pre-fix run and scoped to the method enclosing the change in the pre-fix version rather than to the changed lines themselves, which gives a single well-defined target for insertions, deletions, and updates alike. Coverage in particular is reported as a diagnostic signal, to help the agent notice that its test is exercising the wrong code path, and is never presented to the agent as a score to maximize
5. The agent iteratively improves the test until the candidate compiles and fails on the pre-fix version, or until the retry limit of 5 is reached. Each terminated attempt is classified by its outcome pair: (i) fail / pass is valid; (ii) fail / fail indicates an assertion the fix does not satisfy; (iii) pass on the pre-fix version cannot arise, as it does not meet the loop's stopping criterion.

### 4.3 Role of MCP (Model Context Protocol)

First, a prompt induced pre-loading pipeline is run for fetching and initiating the context of diff (AST or git diff). This context is not a tool itself, but a part of the loop that runs before the agent starts, switches on the condition, and is placed directly into the agent's opening prompt, so that delivery is identical in timing and content across every trial. This step is the way through which information derived from the post-fix version reaches the agent, because the difference of code in the post-fix version is provided to the agent.

A custom MCP server exposes `get_context`, and `run_test` tools, enabling reproducible Defects4J interaction. Both operate over a working directory containing only the pre-fix checkout. `get_context` returns the relevant surrounding code from that checkout and exposes no version parameter or path argument that can resolve outside its root; `run_test` compiles and executes against that checkout. The post-fix checkout is materialized in a separate directory owned by the evaluation harness and is not reachable through any tool on the agent's surface. This isolation is enforced in the server implementation through separate roots and path validation, not by instruction in the prompt.

### 4.4 GumTree and AST Edit Scripts

We use GumTree [1] to generate AST edit scripts (*Insert*, *Delete*, *Update*, *Move*) from code changes. The output is normalized by removing internal IDs and positional metadata, producing concise summaries for LLM input.

### 4.5 Differential Execution Oracle

Tests are validated using a differential execution oracle: a valid regression test must fail on the pre-fix version (due to assertion failure and not due to a generic runtime error) and pass on the post-fix version, ensuring correctness through program behavior rather than LLM assertions. This is the evaluation structure. It is applied once per bug after the agentic loop has terminated, and its result is never returned to the agent.

### 4.6 Evaluation Metrics

The generated tests are evaluated using Fault Diagnosability Gain (FDG) [3], which measures how much a single test improves fault localisation. FDG is not standalone: it is computed over an underlying localisation technique, so we state that setup in full, following the protocol of [3] for comparability.

- **Localisation technique.** We use spectrum-based fault localisation with the Ochiai formula, which scores each line by how strongly its execution correlates with failing tests rather than passing ones. This score is called *suspiciousness*. Line scores are aggregated to method level by taking the most suspicious line, with ties broken by the max tie-breaker.
- **Baseline suite and spectra.** The generated test is added to the failing tests Defects4J [2] provides for that bug. Coverage is measured with Cobertura over the relevant classes, on the pre-fix version only, so no post-fix execution enters the computation.
- **FDG value.** FDG combines two parts: *Split*, which rewards separating program elements that all current tests treat identically, and *Cover*, which rewards executing elements that are already suspicious. We use $\alpha = 0.5$, the default in [3], weighting both equally.
- **Realised localisation gain.** FDG predicts diagnostic value, so we also report whether that prediction holds. We measure the change in acc@n for $n \in \{1, 3, 5, 10\}$, the number of bugs whose faulty method is ranked in the top $n$, and in mean average precision (mAP).
- **Robustness to formula choice.** Different formulas produce different suspiciousness scores, which propagate into both Split and Cover. We recompute all values under DStar [9] ($* = 2$) from the stored spectra, requiring no extra test execution. Ochiai is reported as primary.

FDG and localisation accuracy are computed only for tests satisfying the differential oracle (Section 4.5). The paired comparison is made over bugs where both conditions produced a valid test, and the size of that subset is reported.

---

## 5 Roadmap and Timeline

The project spans from August 31st to December 24th, divided across six key phases based on Fall 2026 deadlines

| Dates | Phase | Key Deliverables |
| :--- | :--- | :--- |
| Aug 31 – Sep 13 | Setup | Select Defects4J commits, implement MCP tools, validate GumTree normalization. |
| Sep 14 – Oct 4 | Baseline | Develop raw-diff agent and validate on pilot commits. |
| Oct 5 – Oct 25 | Experiments | Deploy AST-based agent and evaluate both approaches on the benchmark. |
| Oct 26 – Nov 15 | Analysis | Compute FDG, adequacy, and perform statistical and failure analysis. |
| Nov 16 – Nov 30 | Write-up | Prepare thesis, figures, and final results. Submit thesis to Advisor by Nov 30. |
| Dec 1 – Dec 24 | Evaluation | Thesis review period. Submit evaluation report by Dec 24. |

*Table 2: Project roadmap and timeline based on Fall 2026 graduation research deadlines.*

### 5.1 Anticipated Risks and Fixes

- **Benchmark Availability and Size:** We will confirm a sufficiently large set of Java commits with clean pre/post-fix builds before committing to the full pipeline. The usable set, selected as the conjunction of several filters, may be a small fraction of the starting benchmark, and sufficient for the iterations. Mitigation: we compute the intersection size in the beginning weeks and fix a minimum viable set size before building the agent; if the intersection falls below it, we relax the localized-fix filter first, as it is the least tied to the research question.
- **Compute and API Cost:** Iterative agent loops across many commits can be expensive; we will cap retries per test and may subsample the benchmark if needed.
- **Risk of Losing Data from Parsing Errors:** When we select only the bugs that can successfully parse with the GumTree, there is a possibility that we might lose lots of data. In order to fix this we will keep a careful record of every bug we throw away during data collection. In our final report we will write down the exact number of bugs that were discarded along with the reason why they failed.
- **Data contamination:** Defects4J is likely present in the model's training data, which threatens both attribution and the comparison itself. While the comparison will still assess the change in the quality of the test suite generation, using a fallback using GHRB (GitHub Recent Bugs Dataset) [10] replication allows for an additional reliability scope, which Kang et al.[7, 8] assembled from pull requests merged after model training cutoffs for precisely this reason.

---

## 6 Conclusion

This work investigates whether replacing raw git diffs with GumTree-generated AST edit scripts improves LLM-based regression test generation. By combining an MCP-based agent with a differential execution oracle, test validity is based on actual program behavior rather than LLM output. The study will provide an empirical comparison of textual and structural contexts using FDG, while delivering a reusable MCP-based framework for automated software testing.

---

## Bibliography

- **[1]** Jean-Rémy Falleri, Floréal Morandat, Xavier Blanc, Matias Martinez, and Martin Monperrus. Fine-grained and accurate source code differencing. In *Proceedings of the 29th ACM/IEEE International Conference on Automated Software Engineering*, pages 313–324, 2014. doi: 10.1145/2642937.2642982.
- **[2]** René Just, Darioush Jalali, and Michael D. Ernst. Defects4J: A database of existing faults to enable controlled testing studies for Java programs. In *Proceedings of the 2014 International Symposium on Software Testing and Analysis*, pages 437–440, 2014.
- **[3]** Gabin An and Shin Yoo. FDG: A precise measurement of fault diagnosability gain of test cases. In *ISSTA 2022 - Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis*, pages 14–26, 2022.
- **[4]** Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. AutoCodeRover: Autonomous program improvement. In *Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis*, 2024. doi: 10.1145/3650212.3680384.
- **[5]** Cuong Chi Le, Cuong Duc Van, Minh V. T. Pham, Tung Duy Vu, Hoang N. Phan, Huy N. Phan, and Tien N. Nguyen. TestWeaver: Execution-aware, feedback-driven regression testing generation with large language models, 2025.
- **[6]** Caroline Lemieux, Jeevana Priya Inala, Shuvendu K. Lahiri, and Siddhartha Sen. CodaMosa: Escaping coverage plateaus in test generation with pre-trained large language models. In *Proceedings of the 45th International Conference on Software Engineering*, pages 919–931, 2023. doi: 10.1109/ICSE48619.2023.00085.
- **[7]** Sungmin Kang, Juyeon Yoon, and Shin Yoo. Large language models are few-shot testers: Exploring llm-based general bug reproduction. In *Proceedings of the 45th International Conference on Software Engineering*, pages 2312–2323, 2023. doi: 10.1109/ICSE48619.2023.00194.
- **[8]** Sungmin Kang, Juyeon Yoon, and Shin Yoo. LIBRO: Source code and replication package. https://github.com/coinse/libro, 2023.
- **[9]** W. Eric Wong, Vidroha Debroy, Ruizhi Gao, and Yihao Li. The DStar method for effective software fault localization. *IEEE Transactions on Reliability*, 63(1): 290–308, 2014.
- **[10]** Jae Yong Lee, Sungmin Kang, Juyeon Yoon, and Shin Yoo. The github recent bugs dataset for evaluating llm-based debugging applications, 2023. URL https://arxiv.org/abs/2310.13229.

