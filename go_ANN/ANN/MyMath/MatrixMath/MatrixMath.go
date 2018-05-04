package MatrixMath

// MatrixMult plication of 1xN and NxM
func MatrixMult(OnexN []float32, NxM [][]float32) []float32 {
	thing := make([]float32, len(NxM[0]))

	for i := 0; i < len(NxM[0]); i++ {
		for j := 0; j < len(OnexN); j++ {
			thing[i] += OnexN[j] * NxM[j][i]
		}
	}
	return thing
}
