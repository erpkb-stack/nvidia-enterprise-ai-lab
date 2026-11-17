TECHNOLOGIES = [
 {"name":"CUDA","category":"GPU computing","purpose":"GPU parallel computing platform","when_to_use":"GPU-accelerated compute and low-level runtime integration"},
 {"name":"NVIDIA NIM","category":"Inference","purpose":"Containerized inference microservices with standardized APIs","when_to_use":"Deploy supported foundation models behind production endpoints"},
 {"name":"NeMo Framework","category":"Model development","purpose":"Build, customize, and train generative AI models","when_to_use":"Fine-tuning and model training workflows"},
 {"name":"Nemotron","category":"Models","purpose":"NVIDIA model family for enterprise generative AI","when_to_use":"Evaluate supported variants for assistant and agent workloads"},
 {"name":"Triton Inference Server","category":"Model serving","purpose":"Production serving across supported frameworks and model formats","when_to_use":"Multi-framework serving, ensembles, and high-throughput inference"},
 {"name":"TensorRT","category":"Optimization","purpose":"Optimize deep learning inference for NVIDIA GPUs","when_to_use":"Optimize supported networks for target hardware"},
 {"name":"TensorRT-LLM","category":"LLM inference","purpose":"Optimize and serve supported large language models","when_to_use":"High-performance LLM inference on supported configurations"},
 {"name":"Dynamo","category":"Inference orchestration","purpose":"Distributed inference serving and request orchestration","when_to_use":"Scale distributed inference workloads"},
 {"name":"Run:ai","category":"GPU orchestration","purpose":"GPU workload orchestration and resource management","when_to_use":"Manage shared GPU capacity across teams"},
 {"name":"GPU Operator","category":"Kubernetes","purpose":"Automate NVIDIA GPU software components in Kubernetes","when_to_use":"Standardize GPU enablement on supported clusters"},
 {"name":"Container Toolkit","category":"Containers","purpose":"Expose NVIDIA GPU devices to container workloads","when_to_use":"Run GPU-enabled containers on configured hosts"}
]
USE_CASES = [
 {"id":"rag-assistant","name":"Enterprise Knowledge Assistant","problem":"Grounded answers across policies, manuals, and internal knowledge.","pattern":"RAG","components":["Ingestion","Embeddings","Vector search","Reranking","NIM inference"],"outcome":"Cited answers with access-aware retrieval."},
 {"id":"aml-investigation","name":"AML Investigation Copilot","problem":"Assemble evidence across alerts, entities, and case documents.","pattern":"RAG + tools","components":["Case APIs","Entity resolution","RAG","Agent orchestration","Audit trail"],"outcome":"Faster evidence gathering with human decision authority."},
 {"id":"document-intelligence","name":"Intelligent Document Processing","problem":"Extract structured data from high-volume business documents.","pattern":"Multimodal AI","components":["OCR","Vision model","GPU inference","Human review"],"outcome":"Structured extraction with confidence and review queues."},
 {"id":"manufacturing-vision","name":"Manufacturing Quality Inspection","problem":"Detect defects and anomalies on production lines.","pattern":"Computer vision","components":["Camera ingestion","Vision model","GPU inference","Event integration"],"outcome":"Traceable defect events."},
 {"id":"support-agent","name":"Customer Support Agent","problem":"Resolve support requests using product knowledge and approved actions.","pattern":"RAG + agent tools","components":["Knowledge retrieval","CRM tools","Guardrails","Human escalation"],"outcome":"Grounded responses and controlled automation."}
]
