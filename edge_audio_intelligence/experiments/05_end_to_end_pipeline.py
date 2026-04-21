import os
from edge_audio_intelligence.testbench import SceneConfig, MicArrayConfig, RIRSimulator, PipelineEvaluator
from edge_audio_intelligence.testbench.scene import SourceConfig
from edge_audio_intelligence.pipeline.cascade import CascadePipeline

# Import all modules
from edge_audio_intelligence.modules.ssl.gcc_phat import GccPhatSSL
from edge_audio_intelligence.modules.beamforming.delay_and_sum import DelayAndSumBeamformer
from edge_audio_intelligence.modules.enhancement.spectral_subtraction import SpectralSubtractionEnhancer
from edge_audio_intelligence.modules.asr.whisper_offline import WhisperOfflineASR
from edge_audio_intelligence.modules.diarization.pyannote_diarizer import PyannoteDiarizer

def main():
    print("Setting up acoustic simulation...")
    # 1. Configure the Meeting Room Scene
    config = SceneConfig(
        room_dim=[6.0, 5.0, 3.0],
        rt60=0.3,
        snr_db=15.0, # Realistic meeting noise
        duration_s=10.0,
        sources=[
            # Speaker 0
            SourceConfig(position=[2.0, 3.5, 1.5], signal_type="speech", audio_path="data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac", onset_s=0.0, offset_s=4.0, label="SPEAKER_00", transcription="he hoped there would be stew for dinner"),
            # Speaker 1
            SourceConfig(position=[4.0, 1.5, 1.5], signal_type="speech", audio_path="data/LibriSpeech/test-clean/1212/135807/1212-135807-0000.flac", onset_s=4.5, offset_s=9.0, label="SPEAKER_01", transcription="it was a long journey"),
        ],
        mic_array=MicArrayConfig.smart_speaker_4mic(center=[3.0, 2.5]),
    )

    # Generate the audio scene
    simulator = RIRSimulator(seed=42)
    scene = simulator.generate_scene(config)
    evaluator = PipelineEvaluator(results_dir="results/end_to_end")

    # 2. Build the End-to-End Pipeline
    print("\nBuilding the End-to-End Pipeline...")
    pipeline = CascadePipeline(name="Full_System_V1")
    
    # Stage 1: Locate the sound
    pipeline.add_module(GccPhatSSL())
    
    # Stage 2: Steer microphones towards the sound
    pipeline.add_module(DelayAndSumBeamformer())
    
    # Stage 3: Remove background noise
    pipeline.add_module(SpectralSubtractionEnhancer())
    
    # Stage 4: Diarization ("Who spoke when?") - The module you built!
    pipeline.add_module(PyannoteDiarizer(name="Pyannote_3.1", model_name="pyannote/speaker-diarization-3.1"))
    
    # Stage 5: ASR ("What did they say?")
    pipeline.add_module(WhisperOfflineASR(model_size="tiny"))

    # 3. Run the Pipeline
    print("\nProcessing audio through the pipeline...")
    # This passes the raw audio into Stage 1, and pushes the output through to Stage 5
    final_data = pipeline.run(scene.to_pipeline_dict())

    # 4. Evaluate and Print Results
    print("\nEvaluating all metrics (including DER)...")
    evaluator.run_full_evaluation(final_data, scene, scene_id="Full_Test")
    
    print("\n=== FULL PIPELINE RESULTS ===")
    print(evaluator.print_summary_table())
    
    # Optional: Print out the specific speaker segments your diarizer found
    print("\nDetected Speaker Segments:")
    for start, end, speaker in final_data.get("speaker_segments", []):
        print(f"[{start:.1f}s - {end:.1f}s]: {speaker}")

if __name__ == "__main__":
    main()