; PINNStudio for Windows as an installer (Inno Setup 6): the folder PyInstaller made, dist\PINNStudio\.
;
;   ISCC /DAppVersion=1.6.0 packaging\windows\pinnstudio.iss       (build_windows.bat runs it)
;
; writes dist\PINNStudio-VERSION-setup.exe. It installs for the current user by default, with no administrator
; (the one installing may choose every user instead), adds a Start-menu entry, optionally a desktop shortcut,
; PINNStudio in the "Open with" menu of problem files (.pinn.json; optional), and an uninstaller in Windows'
; list of programs.
;
; Silent:     PINNStudio-VERSION-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER [/DIR=...]
; Uninstall:  "{app}\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES

#ifndef AppVersion
  #error Give the version: ISCC /DAppVersion=X.Y.Z packaging\windows\pinnstudio.iss
#endif
#define AppGuid "B993415D-A9F2-4A53-8B10-C30E7CA0E736"

[Setup]
AppId={{{#AppGuid}}
AppName=PINNStudio
AppVersion={#AppVersion}
AppVerName=PINNStudio {#AppVersion}
AppPublisher=PINNStudio
AppPublisherURL=https://github.com/AsfandyarKhan72/PINNStudio
AppSupportURL=https://github.com/AsfandyarKhan72/PINNStudio/issues
AppUpdatesURL=https://github.com/AsfandyarKhan72/PINNStudio/releases
DefaultDirName={autopf}\PINNStudio
DefaultGroupName=PINNStudio
DisableProgramGroupPage=yes
; for this user, unless the one installing asks for every user (a dialog, or /ALLUSERS)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
SetupIconFile=..\..\pinnstudio\resources\pinnstudio.ico
UninstallDisplayIcon={app}\PINNStudio.exe
UninstallDisplayName=PINNStudio {#AppVersion}
OutputDir=..\..\dist
OutputBaseFilename=PINNStudio-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
ChangesAssociations=yes
; the program cannot be replaced while it runs (or while a training run of it does)
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
Name: "problemfiles"; Description: "Add PINNStudio to ""Open with"" for problem files (.pinn.json)"

[InstallDelete]
; an upgrade replaces the program's libraries whole: files of the version before (another PyTorch, a module
; since removed) must not stay beside the new ones
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\..\dist\PINNStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\PINNStudio"; Filename: "{app}\PINNStudio.exe"; WorkingDir: "{userdocs}"
Name: "{autodesktop}\PINNStudio"; Filename: "{app}\PINNStudio.exe"; WorkingDir: "{userdocs}"; Tasks: desktopicon

[Registry]
; HKA: the current user's classes for a per-user install, the machine's for an install for every user.
; Windows matches the last extension only (.json): taking over every .json file is not PINNStudio's to do, so it
; joins the "Open with" list of .json files instead, through the "PINNStudio.Problem" ProgID
Root: HKA; Subkey: "Software\Classes\PINNStudio.Problem"; ValueType: string; ValueName: ""; \
  ValueData: "PINNStudio problem"; Flags: uninsdeletekey; Tasks: problemfiles
Root: HKA; Subkey: "Software\Classes\PINNStudio.Problem\DefaultIcon"; ValueType: string; ValueName: ""; \
  ValueData: """{app}\PINNStudio.exe"",0"; Tasks: problemfiles
Root: HKA; Subkey: "Software\Classes\PINNStudio.Problem\shell\open\command"; ValueType: string; ValueName: ""; \
  ValueData: """{app}\PINNStudio.exe"" ""%1"""; Tasks: problemfiles
Root: HKA; Subkey: "Software\Classes\.json\OpenWithProgids"; ValueType: string; ValueName: "PINNStudio.Problem"; \
  ValueData: ""; Flags: uninsdeletevalue; Tasks: problemfiles
Root: HKA; Subkey: "Software\Classes\Applications\PINNStudio.exe\shell\open\command"; ValueType: string; \
  ValueName: ""; ValueData: """{app}\PINNStudio.exe"" ""%1"""; Flags: uninsdeletekey; Tasks: problemfiles

[Run]
Filename: "{app}\PINNStudio.exe"; Description: "Start PINNStudio"; Flags: nowait postinstall skipifsilent
