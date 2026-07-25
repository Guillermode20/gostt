// Package clipboard is a tiny wrapper around atotto/clipboard that gives
// the rest of gostt a single import line for both Linux and other platforms
// in future. It also ensures errors carry useful context.
package clipboard

import (
	"errors"
	"time"

	"github.com/atotto/clipboard"
)

// Write deposits s into the system clipboard. Returns a wrapped error if
// the underlying call fails.
//
// We retry a few times because some compositors arbitrate clipboard
// ownership at the point another window gains focus.
func Write(s string) error {
	if s == "" {
		return errors.New("clipboard: empty text")
	}
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		err := clipboard.WriteAll(s)
		if err == nil {
			return nil
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	return lastErr
}

// Read returns the current clipboard contents.
func Read() (string, error) {
	return clipboard.ReadAll()
}
