DEVICE="cpu"
N_WAVEFORMS="1000"
FLOAT64="True"

ripple_time TaylorF2 --device $DEVICE --n-waveforms $N_WAVEFORMS
ripple_time IMRPhenomD --device $DEVICE --n-waveforms $N_WAVEFORMS
ripple_time IMRPhenomXAS --device $DEVICE --n-waveforms $N_WAVEFORMS
ripple_time IMRPhenomPv2 --device $DEVICE --n-waveforms $N_WAVEFORMS