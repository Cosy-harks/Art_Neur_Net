package MyMath

import "math"

// Error returns value between [-2, 2]
func Error(x, y float32) float32 {
	if x == y {
		return 0.0
	}
	// casting nonsense float32 and float64 incompatable
	return (float32)(math.Pow((float64)(x-y), 2.0)) //2 * (x - y) / float32(math.Abs(float64(x))+math.Abs(float64(y)))
}

func dError(x, y float32) float32 {
	if x == y {
		return 0.0
	}
	return 2.0 * (x - y)
}

// AverageError returns average of many errors [-2, 2]
func AverageError(x, y []float32) float32 {
	err := float32(0.0)
	for h := 0; h < len(x); h++ {
		err += Error(x[h], y[h])
	}
	return err / (float32)(len(x))
}

// Average the input slice
func Average(x []float32) float32 {
	var tol float32
	for _, v := range x {
		tol += v
	}
	return tol / (float32)(len(x))
}

// TotalError summation of Errors
func TotalError(x, y []float32) float32 {
	var err float32
	for h := 0; h < len(x); h++ {
		err += Error(x[h], y[h])
	}
	return err
}

//AbsAverage average of absolute values
func AbsAverage(x []float32) float32 {
	var tol float32
	for _, v := range x {
		tol += (float32)(math.Abs((float64)(v)))
	}
	return tol / (float32)(len(x))
}

// Errors returns an array of all the errors
func Errors(x, y []float32) (errs []float32) {
	for e := 0; e < len(x); e++ {
		errs = append(errs, Error(x[e], y[e]))
	}
	return
}

//DeltaErrors returns slice of dErr/dOuts
func DeltaErrors(x, y []float32) (derrs []float32) {
	for e := 0; e < len(x); e++ {
		derrs = append(derrs, dError(x[e], y[e]))
	}
	return
}
