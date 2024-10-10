# Ancient Chinese Machine Reading Comprehension Exception Question Dataset with a Non-trivial Model
Data and code for 'Ancient Chinese Machine Reading Comprehension
Exception Question Dataset with a Non-trivial Model', PRICAI-2023

link: https://link.springer.com/chapter/10.1007/978-981-99-7022-3_14

## ACRE Dataset
The ACRE dataset is in '3_DATASET/ACRC'


### Dataset

|                     | Train | dev | Test | total |
|---------------------|-------|------|------|-------|
| Number of Dataset   | 3286  | 407  | 407  | 4100  |


### Format
The format of ACRE dataset is as follows. 
Sample Attributes
```
{
  "version": "VGaokao-test",  		// Dataset Version
  "data": [
    {
      "cid": 3,                     // Context_id (primary key, a unique tag for a piece of data)
      "context": "诸子之学，兴起于先秦，当时一...", // context
      "dynasty-gpt":"唐朝",          // Calling the dynasty divided by chatgpt
      "dynasty-glm":"唐朝",          // Calling the dynasty divided by chatglm
      "trans_context":"xxxxxxxxxxx"  //Translated into ancient chinese in modern chinese（Using Microsoft azure translator）
      "qas": [
        {
          "qid": "6",               // Question_id 
          "question": "下列...不正确的一项是",   // Question
          "trans_question": "xxxxxxxxxx",   // The question of translating into ancient texts（Using Microsoft azure translator）
          "options": [              // ABCD four options
            "广义上的...",
            "“照着讲...",
            "“接着讲...",
            "不同于以..."
          ],
          "trans_options": [        // Four options for translation into ancient languages
            "广义上的...",
            "“照着讲...",
            "“接着讲...",
            "不同于以..."
          ],
          "answer": "D",            // Answers to the options  
          "correctness": [          // The correctness of the options is irrelevant to the question (an option statement is 1 if it is right and 0 if it is wrong)
            1,
            1,
            1,
            0
          ]
        }
      ]
    }
  ]
}
```

## EVERGREEN model
(EVidence-first bERt encodinG with entiRE-tExt coNvolution)

### python version
* python = 3.8.13
### dependency
#### The versions of important packages are listed here and can be installed according to environment.txt
* torch=1.11.0+cu113
* transformers=4.19.4
* openpyxl=3.1.1
* numpy=1.22.4
* pandas=1.4.3

### How to Run
#### Note: Since the experiments of this model run faster, but there are many different parameter combinations that need to be tested, a batch run mode is added. That is, an excel file is used to configure different parameters to set up the task, and one line in the excel is an experimental task (an experimental task is the complete process of training->validation->testing).
##### So there are two ways to run: (i) run main.py to run a single task; (ii) run run_shell.py to run a batch of multiple tasks.
##### Where should the excel file be placed (you can specify a specific file; you can also specify a directory to get the most recent excel inside the modification time)

## Catalog description

### First-level Catalog
```
1_RUN_EXP              -- Entry point for running experiments
2_DATA_PREPROCESS      -- Dataset processing
3_DATASET              -- Dataset storage
4_RESULT               -- Stores the results of the experiment
test                   -- Test file
utils                  -- For utilities
``` 

### Secondary Catalog - 1_RUN_EXP
```
fewshot_learning                   -- chatgpt/chatglm does fwshot learning, which actually calls the interface
models                             -- Catalog of EVERGREEN models covering multiple branching inputs
models_cnn                         -- The EVERGREEN model is converted to a four-branch input model using multicore convolution and dynamic convolution.
models_compare                     -- Monolithic models for comparison as well as ensemble models
other python file                  -- Code Run Logic
``` 

### Secondary and tertiary catalogs - 2_DATA_PREPROCESS
```
ACRC                             -- ACMRC dataset
      evidence_extract               -- Logic related to evidence extraction
           -- top1_sim1.py           -- The extraction logic that is ultimately used, everything else is an attempt at this process
      extract                        -- Logic for key sentence extraction
      statistics                     -- The logic for the statistics of the dataset
      translate                      -- Translation tools, baidu's translation api and Microsoft Azure's translator
      1_format_data_to_json.py       -- Formatting the raw data
      2_translate.py                 -- Executes the translation logic
      3_context_fwe.py               -- Logic for function word (fwe.txt) deletion
      4_context_extract.py           -- Logic for extracting previous key sentences
      5_data_divide.py               -- The logic of dataset division, that is, according to the different lengths of articles, split into training set, validation set, test set
      6_answer_rebalance.py          -- Option balancing, i.e., to make the number of ABCDs in the training, validation, and test sets as consistent as possible.
      ACRC数据处理逻辑_1.jpg           -- Flowchart of dataset processing
      other                          -- Other related files that may be referenced
other_dataset_format2ACRC        -- Format other datasets into ACMRC dataset format, to facilitate the unification of code logic

``` 

### Secondary and tertiary catalogs - 3_DATASET
```
ACRC             -- ACMRC dataset
        202306        -- June 2024 version, the version of the dataset used for the bios and submissions
             new_data_0504       -- No evidence extraction version added
             top1_sim            -- Evidence extraction added to new_data_0504
                       original         -- evidence extraction with original
                       fwe_context      -- Evidence extraction with text after function word deletion
                       tran_context     -- Evidence extraction with translated text.
                       tran_options     -- Evidence extraction with translated options
        original_data   -- original dataset collected
C3               -- C3 dataset (processed into ACMRC format)
GCRC             -- GCRC dataset (processed into ACMRC format)
NCR              -- NCR dataset (processed into ACMRC format)
VGAOKAO          -- VGAOKAO dataset (processed into ACMRC format)

``` 

### Secondary and tertiary catalogs - 4_RESULT
```
EXCEL file        -- The excel file specified by run_shell.py
Other             -- Stores the results file

``` 

### Secondary and tertiary catalogs - utils
```
Logging utilities, excel read/write utilities, regular utilities, time utilities and other related utilities

``` 

## Experimental code logic

```
main.py (load runtime parameters) -> config.py (wrap and preprocess parameters) -> data_processor.py (convert text to word embeddings) -> data_loader_multiprocess.py (multi-threaded assembly of model's input branches) -> train.py (train) -> predict.py (predict)
``` 

### Description of the input content of the different branches
```
        分支1 : 512（question + options + passage_1）; 

        分支2_1 : 512（question + options + passage_1）+ 512（passage）;

        分支2_2 : 512（question + options）+ 512（passage_1）; 
        
        分支3_1 : 512（question + options）+ 
                              512（passage_1）+ 
                              512（passage_2）;
        
        分支3_2 : 512（question + options + passage_1）+ 
                            512（question + options + passage_2）+ 
                            512（question + options + passage_3）;
                            
        分支4 : 512（question + options_1 + evidence_1（evidence））+ 
               512（question + options_2 + evidence_2（evidence）+
               512（question + options_3 + evidence_3（evidence）+
               512（question + options_4 + evidence_4（evidence）
   
``` 