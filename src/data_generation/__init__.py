"""
Synthetic clinical data generation pipeline.

Generates (transcript, label) pairs for training and evaluation.

Template types:
  - soap      : GP consultation -> SOAP JSON (train)
  - referral_a: GP dictation monologue -> REFERRAL JSON (train)
  - referral_b: GP-patient dialogue about referral -> REFERRAL JSON (zero-shot eval)
  - mse       : Psychiatric intake dialogue -> MSE JSON (zero-shot eval)

Entry point: scripts/gen_synth.py
"""
