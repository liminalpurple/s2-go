package s2_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestS2(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "s2-go test suite")
}
