//go:build !windows

package main

import (
	"os/exec"
	"syscall"
)

// setProcessGroup puts llama-server in its own process group so a stop reaches any children.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func terminate(cmd *exec.Cmd) {
	_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
}
