//go:build cublas

package s2

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcublasLt -lcuda -lnccl
*/
import "C"

func init() {
	defaultBackend = BackendCUDA
}
