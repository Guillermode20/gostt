//go:build !linux && !darwin && !freebsd

package tray

import (
	"log"
	"time"
)

// Run is a no-op stub on unsupported platforms. To keep tests happy it
// still drains t.updateCh into t.title so LastTitle() returns something.
func (t *Tray) Run() error {
	go func() {
		for {
			select {
			case st := <-t.updateCh:
				t.title.Store(st.Title)
				log.Printf("tray (stub): %s", st.Title)
			case <-t.qCh:
				return
			}
		}
	}()
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-t.qCh:
			return nil
		case <-tick.C:
		}
	}
}
