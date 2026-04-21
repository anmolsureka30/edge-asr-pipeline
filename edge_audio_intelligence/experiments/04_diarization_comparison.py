import os
import argparse
from edge_audio_intelligence.testbench import SceneConfig, MicArrayConfig, RIRSimulator, PipelineEvaluator
from edge_audio_intelligence.testbench.scene import SourceConfig
from edge_audio_intelligence.pipeline.cascade import CascadePipeline
from edge_audio_intelligence.modules.ssl.gcc_phat import GccPhatSSL
from edge_audio_intelligence.modules.beamforming.delay_and_sum import DelayAndSumBeamformer
from edge_audio_intelligence.modules.diarization.pyannote_diarizer import PyannoteDiarizer

def main():
    # 1. Configure a multi-speaker scene (simulating a meeting)
    print("Setting up acoustic simulation...")
    config = SceneConfig(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.2, # Low reverb for clarity
        snr_db=20.0,
        duration_s=10.0,
        sources=[
            # Speaker 0 talks from 0.0s to 4.0s
            SourceConfig(position=[2.0, 3.5, 1.5], signal_type="speech", audio_path="data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac", onset_s=0.0, offset_s=4.0, label="SPEAKER_00"),
            # Speaker 1 talks from 4.5s to 9.0s
            SourceConfig(position=[4.0, 1.5, 1.5], signal_type="speech", audio_path="data/LibriSpeech/test-clean/1212/135807/1212-135807-0000.flac", onset_s=4.5, offset_s=9.0, label="SPEAKER_01"),
        ],
        mic_array=MicArrayConfig.smart_speaker_4mic(center=[3.0, 2.5]),
    )

    simulator = RIRSimulator(seed=42)
    scene = simulator.generate_scene(config)
    evaluator = PipelineEvaluator(results_dir="results/diarization")

    # --- Test 1: Pyannote Model 3.1 ---
    pipeline1 = CascadePipeline(name="Diarization_V3.1")
    pipeline1.add_module(GccPhatSSL())
    pipeline1.add_module(DelayAndSumBeamformer())
    pipeline1.add_module(PyannoteDiarizer(name="Pyannote_3.1", model_name="pyannote/speaker-diarization-3.1"))
    
    print("\nRunning Pipeline 1 (Pyannote 3.1)...")
    data1 = pipeline1.run(scene.to_pipeline_dict())
    evaluator.run_full_evaluation(data1, scene, scene_id="Pya_3.1")

    # --- Test 2: Pyannote Model 3.0 ---
    pipeline2 = CascadePipeline(name="Diarization_V3.0")
    pipeline2.add_module(GccPhatSSL())
    pipeline2.add_module(DelayAndSumBeamformer())
    pipeline2.add_module(PyannoteDiarizer(name="Pyannote_3.0", model_name="pyannote/speaker-diarization-3.0"))

    print("\nRunning Pipeline 2 (Pyannote 3.0)...")
    data2 = pipeline2.run(scene.to_pipeline_dict())
    evaluator.run_full_evaluation(data2, scene, scene_id="Pya_3.0")

    print("\n=== DIARIZATION ALGORITHM RANKING ===")
    print("Lower DER (Diarization Error Rate) is better.")
    print(evaluator.print_summary_table())

if __name__ == "__main__":
    main()