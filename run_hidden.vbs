Option Explicit

Dim shell, fso, root, ps, runtime, command, exitCode

Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

root = fso.GetParentFolderName(WScript.ScriptFullName)
ps = shell.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"
runtime = root & "\bridge_runtime.ps1"

shell.CurrentDirectory = root
command = """" & ps & """ -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & runtime & """ -Mode service"
exitCode = shell.Run(command, 0, True)

WScript.Quit exitCode
