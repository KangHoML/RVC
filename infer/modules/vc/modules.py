import traceback
import logging
import os

logger = logging.getLogger(__name__)

import numpy as np
import torch
from io import BytesIO

from infer.lib.audio import load_audio, wav2, save_audio, float_np_array_to_wav_buf
from rvc.synthesizer import get_synthesizer, load_synthesizer
from .info import show_model_info
from .pipeline import Pipeline
from .utils import get_index_path_from_model, load_hubert


class VC:
    def __init__(self, config):
        # 초기화: 모델 관련 변수들을 None으로 설정
        self.n_spk = None          # 화자 수
        self.tgt_sr = None         # 목표 샘플링 레이트
        self.net_g = None          # 생성 모델 네트워크
        self.pipeline = None       # 처리 파이프라인
        self.cpt = None            # 체크포인트 데이터
        self.version = None        # 모델 버전
        self.if_f0 = None          # F0(피치) 사용 여부
        self.version = None        # 버전 정보 (중복됨)
        self.hubert_model = None   # Hubert 특성 추출 모델

        self.config = config       # 설정 저장

    def get_vc(self, sid, *to_return_protect):
        # 모델 ID(sid)를 받아 해당 모델을 로드하는 함수
        logger.info("Get sid: " + sid)
        
        # F0(피치) 보호 관련 UI 가시성 및 값 설정
        to_return_protect0 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[0] if self.if_f0 != 0 and to_return_protect else 0.5
            ),
            "__type__": "update",
        }
        to_return_protect1 = {
            "visible": self.if_f0 != 0,
            "value": (
                to_return_protect[1] if self.if_f0 != 0 and to_return_protect else 0.33
            ),
            "__type__": "update",
        }

        if sid == "" or sid == []:
            # 모델 ID가 비어있는 경우 (모델 언로드)
            if (
                self.hubert_model is not None
            ):  # 폴링을 고려하여, sid가 모델에서 무모델로 전환되었는지 확인하는 판단 추가
                logger.info("Clean model cache")
                # 기존 모델 메모리 정리
                del (self.net_g, self.n_spk, self.hubert_model, self.tgt_sr)  # ,cpt
                self.hubert_model = self.net_g = self.n_spk = self.hubert_model = (
                    self.tgt_sr
                ) = None
                
                # GPU/MPS 메모리 캐시 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
                # 메모리 정리가 제대로 되지 않아 추가 처리
                self.net_g, self.cpt = get_synthesizer(self.cpt, self.config.device)
                self.if_f0 = self.cpt.get("f0", 1)
                self.version = self.cpt.get("version", "v1")
                del self.net_g, self.cpt
                
                # 다시 메모리 캐시 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
            # UI 업데이트를 위한 값 반환
            return (
                (
                    {"visible": False, "__type__": "update"},
                    to_return_protect0,
                    to_return_protect1,
                    {"value": to_return_protect[2], "__type__": "update"},
                    {"value": to_return_protect[3], "__type__": "update"},
                    {"value": "", "__type__": "update"},
                )
                if to_return_protect
                else {"visible": True, "maximum": 0, "__type__": "update"}
            )

        # 모델 파일 경로 설정
        person = f'{os.getenv("weight_root")}/{sid}'
        logger.info(f"Loading: {person}")

        # 모델 로드
        self.net_g, self.cpt = load_synthesizer(person, self.config.device)
        self.tgt_sr = self.cpt["config"][-1]                           # 목표 샘플링 레이트
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # 화자 수
        self.if_f0 = self.cpt.get("f0", 1)                             # F0 사용 여부
        self.version = self.cpt.get("version", "v1")                   # 모델 버전
        
        # 모델 정밀도 설정 (half 또는 float)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
            
        # 파이프라인 초기화
        self.pipeline = Pipeline(self.tgt_sr, self.config)
        
        # 화자 수와 인덱스 정보 설정
        n_spk = self.cpt["config"][-3]
        index = {"value": get_index_path_from_model(sid), "__type__": "update"}
        logger.info("Select index: " + index["value"])
        
        # UI 업데이트를 위한 값 반환
        return (
            (
                {"visible": True, "maximum": n_spk, "__type__": "update"},
                to_return_protect0,
                to_return_protect1,
                index,
                index,
                show_model_info(self.cpt),
            )
            if to_return_protect
            else {"visible": True, "maximum": n_spk, "__type__": "update"}
        )

    def vc_single(
        self,
        sid,                # 모델 ID
        input_audio_path,   # 입력 오디오 경로
        f0_up_key,          # 피치 변경 값
        f0_file,            # F0 파일
        f0_method,          # F0 추출 방법
        file_index,         # 인덱스 파일
        file_index2,        # 대체 인덱스 파일
        index_rate,         # 인덱스 사용 비율
        filter_radius,      # 필터 반경
        resample_sr,        # 리샘플링 샘플링 레이트
        rms_mix_rate,       # RMS 믹스 비율
        protect,            # 보호 값
    ):
        # 단일 오디오 파일을 변환하는 함수
        
        # 입력 파일 확인
        if input_audio_path is None:
            return "You need to upload an audio", None
        elif hasattr(input_audio_path, "name"):
            input_audio_path = str(input_audio_path.name)
            
        f0_up_key = int(f0_up_key)
        
        try:
            # 오디오 로드 및 정규화
            audio = load_audio(input_audio_path, 16000)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                np.divide(audio, audio_max, audio)
                
            times = [0, 0, 0]  # 처리 시간 측정용 (npy, f0, infer)
            
            # Hubert 모델 로드 (필요한 경우)
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config.device, self.config.is_half)
            
            # 인덱스 파일 경로 처리
            if file_index:
                if hasattr(file_index, "name"):
                    file_index = str(file_index.name)
                file_index = (
                    file_index.strip(" ")
                    .strip('"')
                    .strip("\n")
                    .strip('"')
                    .strip(" ")
                    .replace("trained", "added")
                )
            elif file_index2:
                file_index = file_index2
            else:
                file_index = ""  # 잘못된 입력 방지
            
            # 음성 변환 파이프라인 실행
            audio_opt = self.pipeline.pipeline(
                self.hubert_model,
                self.net_g,
                sid,
                audio,
                times,
                f0_up_key,
                f0_method,
                file_index,
                index_rate,
                self.if_f0,
                filter_radius,
                self.tgt_sr,
                resample_sr,
                rms_mix_rate,
                self.version,
                protect,
                f0_file,
            ).astype(np.int16)
            
            # 적절한 타겟 샘플링 레이트 설정
            if self.tgt_sr != resample_sr >= 16000:
                tgt_sr = resample_sr
            else:
                tgt_sr = self.tgt_sr
                
            # 인덱스 사용 정보
            index_info = (
                "Index: %s." % file_index
                if os.path.exists(file_index)
                else "Index not used."
            )
            
            # 성공 메시지와 결과 반환
            return (
                "Success.\n%s\nTime: npy: %.2fs, f0: %.2fs, infer: %.2fs."
                % (index_info, *times),
                (tgt_sr, audio_opt),
            )
            
        except Exception as e:
            # 오류 처리
            info = traceback.format_exc()
            logger.warning(info)
            return str(e), None

    def vc_multi(
        self,
        sid,             # 모델 ID
        dir_path,        # 입력 디렉토리 경로
        opt_root,        # 출력 디렉토리 경로
        paths,           # 입력 파일 경로 목록
        f0_up_key,       # 피치 변경 값
        f0_method,       # F0 추출 방법
        file_index,      # 인덱스 파일
        file_index2,     # 대체 인덱스 파일
        index_rate,      # 인덱스 사용 비율
        filter_radius,   # 필터 반경
        resample_sr,     # 리샘플링 샘플링 레이트
        rms_mix_rate,    # RMS 믹스 비율
        protect,         # 보호 값
        format1,         # 출력 파일 형식
    ):
        # 여러 오디오 파일을 변환하는 함수
        try:
            # 경로 문자열 정리 (공백, 따옴표, 개행 문자 제거)
            dir_path = (
                dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )  # 사용자가 경로를 복사할 때 발생할 수 있는 문제 방지
            opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            
            # 출력 디렉토리 생성
            os.makedirs(opt_root, exist_ok=True)
            
            # 입력 파일 목록 준비
            try:
                if dir_path != "":
                    # 디렉토리 내 모든 파일
                    paths = [
                        os.path.join(dir_path, name) for name in os.listdir(dir_path)
                    ]
                else:
                    # 지정된 파일 목록
                    paths = [path.name for path in paths]
            except:
                traceback.print_exc()
                paths = [path.name for path in paths]
                
            infos = []  # 처리 결과 정보 저장
            
            # 각 파일 처리
            for path in paths:
                # 단일 파일 변환 실행
                info, opt = self.vc_single(
                    sid,
                    path,
                    f0_up_key,
                    None,
                    f0_method,
                    file_index,
                    file_index2,
                    # file_big_npy,  # 주석 처리된 파라미터
                    index_rate,
                    filter_radius,
                    resample_sr,
                    rms_mix_rate,
                    protect,
                )
                
                # 변환 성공 시 파일 저장
                if "Success" in info:
                    try:
                        tgt_sr, audio_opt = opt
                        save_audio(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                            f32=True,
                        )
                    except:
                        info += traceback.format_exc()
                        
                # 결과 정보 추가 및 중간 결과 반환 (제너레이터 사용)
                infos.append("%s->%s" % (os.path.basename(path), info))
                yield "\n".join(infos)
                
            # 최종 결과 반환
            yield "\n".join(infos)
            
        except:
            # 오류 처리
            yield traceback.format_exc()
