"""
简化的单文件图构建脚本
直接在多GPU上并行计算logit分数（softmax归一化后的P(Yes)概率，范围[0,1]），无需client-server架构
"""

import os
import json
import pickle
import threading
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# ============================================================================
# 配置部分
# ============================================================================

# Default values (can be overridden in __main__ based on dataset/target)
CHECK_POINT_DIR = "checkpoint/MIMIC3/HTN2DM/"
CHECK_POINT_PATH = os.path.join(CHECK_POINT_DIR, "graph-checkpoint.pkl")

# ============================================================================
# 模型类
# ============================================================================

class LargeLanguageModel:
    def __init__(
            self,
            model_name_or_path,
            index=0,
            device=None,
            max_context_length=None):
        """
        初始化大语言模型
        
        Args:
            model_name_or_path: 模型路径
            index: 模型索引（用于调试和日志）
            device: 指定GPU设备，如 'cuda:0', 'cuda:1', 'cuda:2'
        """
        self.index = index
        self.device_name = device
        
        print(f"[Model #{self.index}] 正在加载到 {device}...")
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        
        try:
            model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
        except:
            print(f"[Model #{self.index}] 警告: 无法加载generation_config")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Preserve the task instruction at the end when a patient prompt is too
        # long. Left padding is also the correct batching mode for a causal LM.
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"

        length_candidates = [
            getattr(model.config, "max_position_embeddings", None),
            getattr(tokenizer, "model_max_length", None),
        ]
        length_candidates = [
            value
            for value in length_candidates
            if isinstance(value, int) and 0 < value < 1_000_000
        ]
        native_max_length = min(length_candidates) if length_candidates else 2048
        if max_context_length is None:
            self.max_length = native_max_length
        else:
            if max_context_length <= 0:
                raise ValueError("max_context_length must be positive")
            self.max_length = min(max_context_length, native_max_length)
        
        self.model = model
        self.tokenizer = tokenizer
        
        # 获取Yes和No的token IDs
        self.yes_token_id = self._get_token_id("Yes")
        self.no_token_id = self._get_token_id("No")
        
        print(f"[Model #{self.index}] ✓ 加载完成")
        print(f"[Model #{self.index}] Yes token ID: {self.yes_token_id}, No token ID: {self.no_token_id}")
        print(
            f"[Model #{self.index}] Context length: {self.max_length} "
            f"(native maximum: {native_max_length})"
        )
    
    def _get_token_id(self, token: str) -> int:
        """获取指定token的ID"""
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 0:
            raise ValueError(f"Token '{token}' 无法编码")
        return token_ids[0]
    
    def calculate_logit_score(self, prompt: str) -> float:
        """
        计算给定prompt的logit分数
        返回: softmax归一化后的P(Yes)概率，范围 [0, 1]
        """
        model, tokenizer = self.model, self.tokenizer
        
        # 编码prompt
        encodings = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encodings.input_ids.to(model.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 获取最后一个token的logits
        last_token_logits = logits[0, -1, :]  # [vocab_size]
        
        # 提取Yes和No的logits
        logit_yes = last_token_logits[self.yes_token_id].item()
        logit_no = last_token_logits[self.no_token_id].item()
        
        # 使用softmax归一化，得到Yes的概率
        logits_tensor = torch.tensor([logit_yes, logit_no])
        prob_yes = torch.softmax(logits_tensor, dim=0)[0].item()
        
        return prob_yes
    
    def calculate_logit_score_batch(self, prompts: list) -> list:
        """
        批量计算logit分数
        
        Args:
            prompts: prompt列表
            
        Returns:
            logit分数列表 (softmax归一化后的P(Yes)概率，范围 [0, 1])
        """
        model, tokenizer = self.model, self.tokenizer
        
        if len(prompts) == 0:
            return []
        
        # 如果只有一个，直接调用单个计算
        if len(prompts) == 1:
            return [self.calculate_logit_score(prompts[0])]
        
        # 批量编码（padding到相同长度）
        encodings = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        
        # 移动到GPU
        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)
        
        # 批量前向计算
        with torch.no_grad():
            outputs = model(
                input_ids, 
                attention_mask=attention_mask
            )
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # 逐样本提取最后一个有效token的logits
        scores = []
        batch_size = logits.size(0)
        for i in range(batch_size):
            # 找到最后一个有效token的位置（非padding）
            valid_positions = attention_mask[i].nonzero(as_tuple=False)
            if valid_positions.numel() == 0:
                raise ValueError(f"Prompt {i} contains no valid tokens")
            last_valid_pos = valid_positions[-1].item()
            
            # 获取最后一个有效token的logits
            last_token_logits = logits[i, last_valid_pos, :]  # [vocab_size]
            
            # 提取Yes和No的logits
            logit_yes = last_token_logits[self.yes_token_id].item()
            logit_no = last_token_logits[self.no_token_id].item()
            
            # 使用softmax归一化，得到Yes的概率
            logits_tensor = torch.tensor([logit_yes, logit_no])
            prob_yes = torch.softmax(logits_tensor, dim=0)[0].item()
            scores.append(prob_yes)
        
        return scores
    
    def calculate_perplexity_for_text(self, prompt: str, text: str) -> float:
        """计算给定prompt下text的perplexity"""
        model, tokenizer = self.model, self.tokenizer
        
        # 编码完整文本
        full_text = prompt + text
        encodings = tokenizer(full_text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        max_length = getattr(model.config, 'model_max_length', getattr(model.config, 'max_position_embeddings', 2048))
        
        # 计算prompt长度
        prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1) - 1
        prompt_len = max(prompt_len, 0)
        
        # 定位结束位置
        end_loc = min(prompt_len + max_length, seq_len)
        target_len = end_loc - prompt_len
        begin_loc = max(end_loc - max_length, 0)
        
        # 移动到模型设备
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100
        
        # 计算perplexity
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        ppl = torch.exp(neg_log_likelihood)
        return ppl.item()
    
    def calculate_perplexity_batch(self, prompts: list, texts: list) -> list:
        """
        批量计算perplexity，真正的批处理实现（一次前向）
        
        Args:
            prompts: prompt列表
            texts: text列表
            
        Returns:
            perplexity值列表
        """
        model, tokenizer = self.model, self.tokenizer
        
        if len(prompts) != len(texts):
            raise ValueError(f"prompts和texts长度不匹配: {len(prompts)} vs {len(texts)}")
        
        if len(prompts) == 0:
            return []
        
        # 如果只有一个，直接调用单个计算
        if len(prompts) == 1:
            return [self.calculate_perplexity_for_text(prompts[0], texts[0])]
        
        # 构建完整文本
        full_texts = [p + t for p, t in zip(prompts, texts)]
        
        # 获取模型最大长度
        max_length = getattr(model.config, 'model_max_length', getattr(model.config, 'max_position_embeddings', 2048))
        
        # 批量编码（padding到相同长度）
        encodings = tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # 移动到GPU
        input_ids = encodings.input_ids.to(model.device)
        attention_mask = encodings.attention_mask.to(model.device)
        
        # 计算每个样本的prompt长度
        prompt_lens = []
        for prompt in prompts:
            prompt_encoding = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_encoding.input_ids.size(1) - 1
            prompt_lens.append(max(prompt_len, 0))
        
        # 创建labels，mask掉prompt部分和padding部分
        labels = input_ids.clone()
        for i, prompt_len in enumerate(prompt_lens):
            # mask掉prompt部分
            labels[i, :prompt_len] = -100
            # mask掉padding部分
            labels[i, attention_mask[i] == 0] = -100
        
        # **关键：一次性批量前向计算所有样本**
        with torch.no_grad():
            outputs = model(
                input_ids, 
                attention_mask=attention_mask
            )
            
            # 获取logits: [batch_size, seq_len, vocab_size]
            logits = outputs.logits
            
            # 手动计算每个样本的loss和perplexity
            ppls = []
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            
            # shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
            shift_labels = labels[..., 1:].contiguous()       # [batch, seq_len-1]
            
            # 逐样本计算loss（但所有样本的logits已经批量计算好了）
            batch_size = shift_logits.size(0)
            for i in range(batch_size):
                # 获取当前样本的logits和labels
                sample_logits = shift_logits[i]      # [seq_len-1, vocab]
                sample_labels = shift_labels[i]      # [seq_len-1]
                
                # 计算loss（只计算非-100的位置）
                losses = loss_fct(sample_logits, sample_labels)  # [seq_len-1]
                
                # 只对有效token计算平均loss
                valid_mask = (sample_labels != -100)
                if valid_mask.sum() > 0:
                    avg_loss = losses[valid_mask].mean()
                    ppl = torch.exp(avg_loss)
                    ppls.append(ppl.item())
                else:
                    # 如果没有有效token，返回一个很大的perplexity
                    ppls.append(float('inf'))
        
        return ppls

# ============================================================================
# 模型池
# ============================================================================

class ModelPool:
    def __init__(
            self,
            model_name_or_path,
            n_models=3,
            gpu_ids=None,
            max_context_length=None):
        """
        初始化模型池，支持多GPU
        
        Args:
            model_name_or_path: 模型路径
            n_models: 模型数量
            gpu_ids: GPU设备ID列表，如 [0, 1, 2]
        """
        self.n_models = n_models
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        
        print(f"\n{'='*80}")
        print(f"初始化模型池: {n_models} 个模型")
        print(f"使用GPU: {self.gpu_ids}")
        print(f"模型分配策略: 轮询分配")
        print(f"{'='*80}\n")
        
        # 轮询分配模型到GPU
        self.models = []
        for idx in range(n_models):
            gpu_id = self.gpu_ids[idx % len(self.gpu_ids)]
            device = f"cuda:{gpu_id}"
            model = LargeLanguageModel(
                model_name_or_path,
                index=idx,
                device=device,
                max_context_length=max_context_length,
            )
            self.models.append(model)
        
        # 打印GPU显存使用情况
        if torch.cuda.is_available():
            print("\nGPU显存使用情况:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {allocated:.2f}GB / {total:.2f}GB")
            print()
        
        self.lock = threading.Lock()
        self.semaphore = threading.Semaphore(n_models)
    
    def acquire(self):
        """获取一个可用模型"""
        self.semaphore.acquire()
        self.lock.acquire()
        model = self.models.pop()
        self.lock.release()
        return model
    
    def release(self, model):
        """释放模型回池中"""
        self.lock.acquire()
        self.models.append(model)
        self.lock.release()
        self.semaphore.release()

# ============================================================================
# 主处理类
# ============================================================================

class GraphBuilder:
    def __init__(
            self, 
            model_name_or_path,
            data_path, 
            code_map_path,
            output_path,
            text_template=None,
            visit_data_path=None,
            patient_visit_template=None,
            patient_task_template=None,
            max_context_length=None,
            n_models=3,
            gpu_ids=None,
            num_workers=None,
            batch_size=16):
        """
        图构建器
        
        Args:
            model_name_or_path: 模型路径
            data_path: 患者诊断数据路径
            code_map_path: 疾病代码映射路径
            output_path: 输出图数据路径
            text_template: 旧版单段 prompt 模板（仅用于向后兼容）
            visit_data_path: 按患者保存的格式化就诊历史路径
            patient_visit_template: Patient Visit History 模板
            patient_task_template: 疾病对判断任务模板
            max_context_length: 最大上下文长度（不会超过模型原生上限）
            n_models: 模型数量
            gpu_ids: GPU ID列表
            num_workers: 线程池大小（默认等于模型数量）
            batch_size: 批处理大小（每次送多少个疾病对到GPU）
        """
        # 加载数据
        with open(data_path, "r") as f:
            self.diagnoses: dict = json.load(f)
        
        # 加载疾病代码映射
        with open(code_map_path, "r") as f:
            self.disease_id_map = json.load(f)

        self.visit_histories = None
        if visit_data_path is not None:
            with open(visit_data_path, "r") as f:
                self.visit_histories = json.load(f)
        
        self.n_disease = len(self.disease_id_map)
        self.output_path = output_path
        self.text_template = text_template
        self.patient_visit_template = patient_visit_template
        self.patient_task_template = patient_task_template
        self.batch_size = batch_size

        uses_patient_prompt = (
            self.patient_visit_template is not None
            or self.patient_task_template is not None
        )
        if uses_patient_prompt and (
            self.patient_visit_template is None
            or self.patient_task_template is None
            or self.visit_histories is None
        ):
            raise ValueError(
                "Patient-specific prompting requires visit_data_path, "
                "patient_visit_template, and patient_task_template."
            )
        if not uses_patient_prompt and self.text_template is None:
            raise ValueError("A prompt template must be provided.")
        
        # 加载checkpoint（如果存在）
        if os.path.exists(CHECK_POINT_PATH):
            with open(CHECK_POINT_PATH, "rb") as f:
                self.patient_graph_map = pickle.load(f)
                # 找到最后一个有数据的患者ID
                patient_ids = list(self.patient_graph_map.keys())
                self.last_patient_id = patient_ids[-1] if patient_ids else 0
                
                # 统计已有边的情况
                total_edges = sum(len(edges) for edges in self.patient_graph_map.values())
                patients_with_edges = sum(1 for edges in self.patient_graph_map.values() if len(edges) > 0)
                empty_patients = len(self.patient_graph_map) - patients_with_edges
                
                print(f"\n从checkpoint加载:")
                print(f"  文件: {CHECK_POINT_PATH}")
                print(f"  总患者数: {len(self.patient_graph_map)}")
                print(f"  有边的患者: {patients_with_edges}")
                print(f"  空患者: {empty_patients}")
                print(f"  总边数: {total_edges}")
                
                if total_edges == 0 and len(self.patient_graph_map) > 0:
                    print(f"\n⚠️  警告: checkpoint中有 {len(self.patient_graph_map)} 个患者但没有任何边!")
                    print(f"    这可能是之前运行时出现了问题")
                    user_input = input(f"    是否要清空checkpoint重新开始? (y/n): ")
                    if user_input.lower() == 'y':
                        self.patient_graph_map = {}
                        self.last_patient_id = 0
                        print(f"    已清空，将从头开始处理")
                    else:
                        print(f"    将继续处理（跳过已有患者）")
                elif total_edges > 0:
                    avg_edges = total_edges / patients_with_edges if patients_with_edges > 0 else 0
                    print(f"  平均每个患者边数: {avg_edges:.2f}")
                    print(f"  将跳过已处理的患者，继续处理")
        else:
            self.patient_graph_map = {}
            self.last_patient_id = 0
            print(f"\n未找到checkpoint，从头开始")
            print(f"  将创建新的checkpoint: {CHECK_POINT_PATH}")
        
        # 初始化模型池
        self.model_pool = ModelPool(
            model_name_or_path,
            n_models=n_models,
            gpu_ids=gpu_ids,
            max_context_length=max_context_length,
        )
        
        # 线程池设置
        self.num_workers = num_workers if num_workers else n_models
        self.lock = threading.Lock()
        
        print(f"批处理大小: {batch_size}")
        print(f"预期加速: {batch_size}x (相比逐个处理)\n")
    
    def _build_prompt(self, patient_id: int, disease1: str, disease2: str) -> str:
        """Build the two-stage patient-specific prompt used for PCS scoring."""
        if self.patient_visit_template is None:
            return self.text_template.format(disease1, disease2)

        patient_key = str(patient_id)
        if patient_key not in self.visit_histories:
            raise KeyError(f"Patient {patient_id} has no visit history")

        visit_prompt = self.patient_visit_template.format(
            VISIT_LIST=self.visit_histories[patient_key]
        )
        task_prompt = self.patient_task_template.format(
            DISEASE_A=disease1,
            DISEASE_B=disease2,
        )
        return visit_prompt + "\n" + task_prompt

    def run(self):
        """主运行函数 - 批处理版本"""
        print(f"\n开始处理 {len(self.diagnoses)} 个患者...")
        print(f"使用 {self.num_workers} 个工作线程\n")
        
        for patient_id, diseases in tqdm(self.diagnoses.items(), desc="处理患者"):
            patient_id = int(patient_id)
            
            # 跳过已处理的患者
            if patient_id in self.patient_graph_map:
                # 如果这个患者已经有边了，跳过
                if len(self.patient_graph_map[patient_id]) > 0:
                    continue
            
            # 限制疾病数量（避免过多疾病对）
            diseases = diseases[:10] if len(diseases) > 10 else diseases
            
            if len(diseases) <= 1:
                # 没有疾病对需要计算
                self.patient_graph_map[patient_id] = []
                continue
            
            # 收集所有疾病对
            disease_pairs = []
            for disease1 in diseases:
                for disease2 in diseases:
                    if disease1 == disease2:
                        continue
                    disease_pairs.append((disease1, disease2))
            
            if len(disease_pairs) == 0:
                self.patient_graph_map[patient_id] = []
                continue
            
            # 批处理计算
            self.edges = []
            pool = ThreadPoolExecutor(max_workers=self.num_workers)
            futures = []
            
            # 将疾病对分批
            for i in range(0, len(disease_pairs), self.batch_size):
                batch = disease_pairs[i:i+self.batch_size]
                future = pool.submit(self.calculate_batch, patient_id, batch)
                futures.append(future)
            
            pool.shutdown(wait=True)
            
            # 检查错误
            failed_count = 0
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    if failed_count == 0:
                        print(f"\nERROR in patient {patient_id}: {e}")
                    failed_count += 1
            
            if failed_count > 0:
                # Never persist a partially constructed patient graph. Save all
                # previously completed patients, then stop so this patient is
                # retried in full on the next run.
                self.edges = []
                self.save_graph(CHECK_POINT_PATH)
                raise RuntimeError(
                    f"Patient {patient_id} failed in {failed_count} batch(es); "
                    "its partial graph was discarded."
                )
            
            # 保存结果
            self.patient_graph_map[patient_id] = self.edges
            
            # 打印当前患者的处理结果
            # if len(self.patient_graph_map) % 1 == 0:  # 每个患者都打印
            #     edges_count = len(self.edges)
            #     if edges_count > 0:
            #         # 显示第一条边作为示例
            #         sample_edge = self.edges[0]
            #         print(f"  患者 {patient_id}: {edges_count} 条边, 示例边: ({sample_edge[0]}, {sample_edge[1]}, score={sample_edge[2]:.4f})")
            #     else:
            #         print(f"  患者 {patient_id}: 0 条边 (疾病数: {len(diseases)})")
            
            # 定期保存checkpoint
            if len(self.patient_graph_map) % 10 == 0:
                self.save_graph(CHECK_POINT_PATH)
            
            # 每50个患者保存一次独立文件
            if len(self.patient_graph_map) % 50 == 0:
                self.save_graph(CHECK_POINT_DIR + f"graph-{patient_id}.pkl")
        
        # 最终保存
        print(f"\n处理完成! 共 {len(self.patient_graph_map)} 个患者")
        self.save_graph(self.output_path)
        print(f"结果已保存到: {self.output_path}")
    
    def calculate_batch(self, patient_id: int, disease_pairs: list):
        """
        批量计算疾病对的logit分数（P(Yes)概率）
        
        Args:
            patient_id: 患者ID
            disease_pairs: 疾病对列表 [(disease1, disease2), ...]
        """
        try:
            # 准备批处理数据
            prompts = []
            disease_ids = []
            
            for disease1, disease2 in disease_pairs:
                # 检查疾病代码是否存在
                if disease1 not in self.disease_id_map:
                    print(f"警告: disease '{disease1}' not in disease_id_map")
                    continue
                if disease2 not in self.disease_id_map:
                    print(f"警告: disease '{disease2}' not in disease_id_map")
                    continue
                
                disease1_id = self.disease_id_map[disease1]
                disease2_id = self.disease_id_map[disease2]
                
                # 构建包含患者就诊历史和疾病对任务的两阶段 prompt
                prompt = self._build_prompt(patient_id, disease1, disease2)
                prompts.append(prompt)
                disease_ids.append((disease1_id, disease2_id))
            
            if len(prompts) == 0:
                return
            
            # 从模型池获取模型
            model = self.model_pool.acquire()
            try:
                # 批量计算logit分数
                results = model.calculate_logit_score_batch(prompts)
            finally:
                self.model_pool.release(model)
            
            # 保存结果
            self.lock.acquire()
            for (disease1_id, disease2_id), result in zip(disease_ids, results):
                self.edges.append((disease1_id, disease2_id, result))
            self.lock.release()
            
        except Exception as e:
            raise Exception(f"calculate_batch failed: {e}") from e
    
    def calculate(self, patient_id: int, disease1: str, disease2: str):
        """计算单个疾病对的logit分数（P(Yes)概率，保留用于兼容）"""
        try:
            # 检查疾病代码是否存在
            if disease1 not in self.disease_id_map:
                raise KeyError(f"disease1 '{disease1}' not in disease_id_map")
            if disease2 not in self.disease_id_map:
                raise KeyError(f"disease2 '{disease2}' not in disease_id_map")
            
            disease1_id = self.disease_id_map[disease1]
            disease2_id = self.disease_id_map[disease2]
            
            # 构建包含患者就诊历史和疾病对任务的两阶段 prompt
            prompt = self._build_prompt(patient_id, disease1, disease2)
            
            # 从模型池获取模型
            model = self.model_pool.acquire()
            try:
                result = model.calculate_logit_score(prompt)
            finally:
                self.model_pool.release(model)
            
            # 保存结果
            self.lock.acquire()
            self.edges.append((disease1_id, disease2_id, result))
            self.lock.release()
            
        except Exception as e:
            raise Exception(f"calculate failed for {disease1}->{disease2}: {e}") from e
    
    def save_graph(self, path="graph.pkl"):
        """保存图数据（带数据检查）"""
        # 检查数据完整性
        total_patients = len(self.patient_graph_map)
        total_edges = sum(len(edges) for edges in self.patient_graph_map.values())
        patients_with_edges = sum(1 for edges in self.patient_graph_map.values() if len(edges) > 0)
        empty_patients = total_patients - patients_with_edges
        
        print(f"\n{'='*60}")
        print(f"准备保存到: {path}")
        print(f"数据统计:")
        print(f"  总患者数: {total_patients}")
        print(f"  有边的患者: {patients_with_edges}")
        print(f"  空患者: {empty_patients}")
        print(f"  总边数: {total_edges}")
        
        if total_edges > 0:
            avg_edges = total_edges / patients_with_edges if patients_with_edges > 0 else 0
            print(f"  平均每个患者边数: {avg_edges:.2f}")
            
            # 显示几个样例
            sample_patients = [(pid, edges) for pid, edges in self.patient_graph_map.items() if len(edges) > 0][:3]
            if sample_patients:
                print(f"\n样例数据（前3个有边的患者）:")
                for patient_id, edges in sample_patients:
                    print(f"  患者 {patient_id}: {len(edges)} 条边")
                    # 显示第一条边的详细信息
                    edge = edges[0]
                    print(f"    示例边: disease_id={edge[0]} -> disease_id={edge[1]}, score={edge[2]:.4f}")
        else:
            print(f"\n⚠️  警告: 没有任何边数据！")
            print(f"    请检查是否所有患者的疾病数都<=1")
        
        print(f"{'='*60}\n")
        
        # 保存数据
        with open(path, "wb") as f:
            pickle.dump(self.patient_graph_map, f)
        
        print(f"✓ 已保存到: {path}\n")

# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import os
    import sys
    import argparse
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import (
        LLM_MODEL_NAME,
        LLM_MODEL_PATH,
        LLM_MODEL_NUM,
        LLM_GPU_IDS,
        LLM_MAX_CONTEXT_LENGTH,
        PATIENT_VISIT_PROMPT_TEMPLATE,
        PATIENT_TASK_PROMPT_TEMPLATE,
    )
    from data_preprocess.data_config import DataConfig
    
    parser = argparse.ArgumentParser(description="Build patient graph with logit scoring (task-aware)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MIMIC3",
        choices=["MIMIC3", "MIMIC4", "HuaDong"],
        help="Dataset name (default: MIMIC3)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="dm",
        choices=["dm", "cvd", "ckd", "DM", "CVD", "CKD"],
        help="Comorbidity task target: dm/cvd/ckd (default: dm)",
    )
    parser.add_argument(
        "--output-graph-path",
        type=str,
        default=None,
        help="Override output graph path (default: data/<dataset>/HTN2<TARGET>/graph_<model>.pkl)",
    )
    args = parser.parse_args()

    data_config = DataConfig(args.dataset, target=args.target)
    patient_dataset_path = data_config.patient_dataset_path
    icd_to_id_map_path = data_config.icd_to_id_map_path

    # Put checkpoint under the corresponding task data folder to avoid cross-task collisions
    CHECK_POINT_DIR = os.path.join(data_config.output_path, "graph_checkpoints")
    CHECK_POINT_PATH = os.path.join(
        CHECK_POINT_DIR, "graph-checkpoint-patient-prompt.pkl"
    )
    
    # Output graph goes into the corresponding task data folder by default
    graph_dataset_path = (
        args.output_graph_path
        if args.output_graph_path
        else os.path.join(data_config.output_path, f"graph_{LLM_MODEL_NAME}.pkl")
    )
    
    # 创建checkpoint目录
    if not os.path.exists(CHECK_POINT_DIR):
        os.makedirs(CHECK_POINT_DIR)
    
    # 批处理大小 - 根据GPU显存调整
    # 推荐: 24GB显存可以设置为16-32
    BATCH_SIZE = 64
    
    # 构建图
    builder = GraphBuilder(
        model_name_or_path=LLM_MODEL_PATH,
        data_path=patient_dataset_path,
        code_map_path=icd_to_id_map_path,
        output_path=graph_dataset_path,
        visit_data_path=data_config.trunced_ehr_data_path,
        patient_visit_template=PATIENT_VISIT_PROMPT_TEMPLATE,
        patient_task_template=PATIENT_TASK_PROMPT_TEMPLATE,
        max_context_length=LLM_MAX_CONTEXT_LENGTH,
        n_models=LLM_MODEL_NUM,
        gpu_ids=LLM_GPU_IDS,
        num_workers=LLM_MODEL_NUM,
        batch_size=BATCH_SIZE
    )
    
    builder.run()

