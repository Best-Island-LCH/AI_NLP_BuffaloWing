import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser   

from peft import PeftModel

class Pipeline:
    def __init__(
        self,
        model_name: str = "kakaocorp/kanana-1.5-8b-instruct-2505",
        adapter_name: str = "jinn33/kanana-1.5-8b-rlhf",
        max_new_tokens: int = 128,
        do_sample: bool = False,
    ):
        # 1. 토크나이저 로드
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 2. Base 모델 로드
        print(f"📦 Base 모델 로드: {self.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 3. PEFT 어댑터 로드 및 병합
        print(f"🔧 RLHF 어댑터 로드: {adapter_name}")
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).eval()
        
        print("✅ 모델 로드 완료")

        #2.허깅페이스 랭체인 파이프라인 
        self.gen_pipe= pipeline(
             task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            return_full_text=True,  
        )   

        self.hf_llm = HuggingFacePipeline(pipeline=self.gen_pipe)

        #3.프롬프트 
        self.answer_style_guide ="""
[ANSWER STYLE]
- 답변은 가능하면 "도입 1문장 + 번호 목록(3~7개) + 마무리 1문장" 구조를 사용한다.
- 번호 목록 각 항목은 "키워드: 1~2문장 설명" 형식으로 작성한다.
- 톤은 중립적으로 "~할 수 있다/고려해야 한다"를 사용한다.
- 답변은 최대 10줄(또는 800자) 이내로 작성한다.
- 단, 질문이 숫자/연도/인물/계산 등 단답형이면 목록 없이 한 문장으로 답한다.
- 말투는 예의 있고 존댓말을 사용한다.
""".strip()
        self.system_policy=f"""
너는 일반 지식 질문에 답하는 챗봇이다.
아래 규칙과 스타일을 반드시 따른다.    

[SAFETY / UNCERTAINTY]
- 확실한 근거가 없으면 "Insufficient information"이라고만 답한다. (추가 설명 금지)
- 예측/미래/확률적 질문(“~할까요?”)에서 확정 근거가 없으면 "Insufficient information"을 우선한다.
- 근거가 불충분한 일반론만 말하게 될 경우, 답변은 "Insufficient information"을 우선한다.

{self.answer_style_guide}
""".strip()  


        self.fewshot_examples="""
[EXAMPLES]
User Question: 자동차 배기가스의 위험은 무엇인가?
Assistant:
자동차 배기가스는 환경과 건강에 여러 부정적 영향을 줄 수 있습니다.
1. 대기 오염: 미세먼지·질소산화물 등으로 공기 질이 나빠질 수 있습니다.
2. 건강 영향: 호흡기·눈·코 자극을 유발할 수 있고, 장기 노출 시 위험이 커질 수 있습니다.
3. 기후 변화: 온실가스 배출로 지구 온난화에 영향을 줄 수 있습니다.
4. 오존 생성: 광화학 반응으로 오존이 늘어나 호흡기 증상을 악화시킬 수 있습니다.
따라서 배출 저감과 노출을 줄이는 노력이 필요할 수 있습니다.
""".strip()
        
        self.prompt_tmpl = PromptTemplate.from_template(
    """{system_policy}

{fewshot_examples}

[User Question]
{user_question}
""".strip()
)
        
        
        self.retry_prompt_tmpl = PromptTemplate.from_template(
            """{system_policy}

너의 직전 출력이 규칙/스타일을 충분히 지키지 못했다.
이번에는 반드시 규칙을 지키고 다시 답하라.

{fewshot_examples}

[User Question]
{user_question}

[Your previous output]
{previous_output}
""".strip()
        )  
        #4.체인구성
        self.chain =(
            self.prompt_tmpl
            | RunnableLambda(self._to_exaone_chatprompt)
            | RunnableLambda(self._run_hf)
            | StrOutputParser()
        ) 
        self.raw_chain = (
            self.prompt_tmpl
            | RunnableLambda(self._to_exaone_chatprompt)
            | RunnableLambda(self._run_hf)
            | StrOutputParser()
        )

        self.retry_chain = (
            self.retry_prompt_tmpl
            | RunnableLambda(self._to_exaone_chatprompt)
            | RunnableLambda(self._run_hf)
            | StrOutputParser()
        )

        #내부 메서드 : 1) LC prompt -> EXAONE chat template 문자열
    def _to_exaone_chatprompt(self, prompt_value):  #랭체인 프롬프트 문자열을 엑사원 문자열로 바꾸기.  
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        messages = [{"role": "user", "content": text}]
        chat_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return {"chat_prompt": chat_prompt}
        
        #내부 메서드 : 2). HF 호출 + echo 제거
    def _run_hf(self, d):
        chat_prompt = d["chat_prompt"]
        out = self.hf_llm.invoke(chat_prompt)

        if isinstance(out, str) and out.startswith(chat_prompt):   
            out = out[len(chat_prompt):]
        return out.strip()   
        
        #외부 공개 메서드: generate() 
    def generate(self, message: str) -> str:
        inputs = {
            "system_policy": self.system_policy,
            "fewshot_examples": self.fewshot_examples,
            "user_question": message,
        }
             # 1차 시도
        try:
            return self.chain.invoke(inputs)
        except Exception:
            # previous_output 확보
            try:  
                previous_output = self.raw_chain.invoke(inputs)
            except Exception:
                previous_output = ""     

            retry_inputs = {
                "system_policy": self.system_policy,
                "fewshot_examples": self.fewshot_examples,
                "user_question": message,
                "previous_output": previous_output,
            }
                # 재시도 1회
            try:
                return self.retry_chain.invoke(retry_inputs)
            except Exception:
                return "죄송합니다. 다시 시도해주세요."  
                

  
        



