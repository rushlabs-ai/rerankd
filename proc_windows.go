//go:build windows

package main

import "os/exec"

func setProcessGroup(cmd *exec.Cmd) {}

func terminate(cmd *exec.Cmd) {
	_ = cmd.Process.Kill()
}
