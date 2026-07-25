package ui

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"

	_ "embed"
)

//go:embed FiraCode-Regular.ttf
var firaCodeFont []byte

// monospaceResource wraps raw bytes as a Fyne Resource.
type monospaceResource struct{ data []byte }

func (m *monospaceResource) Name() string      { return "FiraCode-Regular.ttf" }
func (m *monospaceResource) Content() []byte    { return m.data }
func (m *monospaceResource) CacheID() string    { return "fira-code-regular" }
func (m *monospaceResource) Reader() *fyne.StaticResource {
	return &fyne.StaticResource{StaticName: m.Name(), StaticContent: m.data}
}

// DarkMonospace is a compact dark theme with Fira Code and zero-radius sharp corners.
type DarkMonospace struct {
	font fyne.Resource
}

func NewDarkMonospace() fyne.Theme {
	return &DarkMonospace{font: &monospaceResource{data: firaCodeFont}}
}

// Color palette — dark terminal aesthetic.
var (
	bgDark    = color.NRGBA{R: 13, G: 17, B: 23, A: 255}   // #0d1117
	fgLight   = color.NRGBA{R: 201, G: 209, B: 217, A: 255} // #c9d1d9
	accent    = color.NRGBA{R: 0, G: 255, B: 65, A: 255}    // #00ff41
	btnBg     = color.NRGBA{R: 33, G: 38, B: 45, A: 255}    // #21262d
	btnHover  = color.NRGBA{R: 48, G: 54, B: 61, A: 255}    // #30363d
	inputBg   = color.NRGBA{R: 22, G: 27, B: 34, A: 255}    // #161b22
	inputBord = color.NRGBA{R: 48, G: 54, B: 61, A: 255}    // #30363d
	selBg     = color.NRGBA{R: 0, G: 80, B: 200, A: 255}    // #0050c8
	placeH    = color.NRGBA{R: 110, G: 118, B: 129, A: 255} // #6e7681
	dimFg     = color.NRGBA{R: 139, G: 148, B: 158, A: 255} // #8b949e
	errRed    = color.NRGBA{R: 255, G: 85, B: 85, A: 255}   // #ff5555
	success   = color.NRGBA{R: 63, G: 185, B: 80, A: 255}   // #3fb950
	warnYel   = color.NRGBA{R: 210, G: 153, B: 34, A: 255}  // #d29922
	separator = color.NRGBA{R: 48, G: 54, B: 61, A: 255}    // #30363d
	shadow    = color.NRGBA{R: 0, G: 0, B: 0, A: 80}
	scrollBg  = color.NRGBA{R: 22, G: 27, B: 34, A: 100}
	scrollFg  = color.NRGBA{R: 110, G: 118, B: 129, A: 180}
)

func (d *DarkMonospace) Color(name fyne.ThemeColorName, _ fyne.ThemeVariant) color.Color {
	switch name {
	case theme.ColorNameBackground, theme.ColorNameOverlayBackground:
		return bgDark
	case theme.ColorNameButton:
		return btnBg
	case theme.ColorNameDisabledButton:
		return btnBg
	case theme.ColorNameDisabled:
		return dimFg
	case theme.ColorNameError:
		return errRed
	case theme.ColorNameFocus:
		return accent
	case theme.ColorNameForeground:
		return fgLight
	case theme.ColorNameForegroundOnError:
		return fgLight
	case theme.ColorNameForegroundOnPrimary:
		return bgDark
	case theme.ColorNameForegroundOnSuccess:
		return bgDark
	case theme.ColorNameForegroundOnWarning:
		return bgDark
	case theme.ColorNameHeaderBackground:
		return color.NRGBA{R: 13, G: 17, B: 23, A: 240}
	case theme.ColorNameHover:
		return btnHover
	case theme.ColorNameHyperlink:
		return accent
	case theme.ColorNameInnerWindowBorder:
		return separator
	case theme.ColorNameInnerWindowBorderInactive:
		return separator
	case theme.ColorNameInputBackground:
		return inputBg
	case theme.ColorNameInputBorder:
		return inputBord
	case theme.ColorNameMenuBackground:
		return color.NRGBA{R: 22, G: 27, B: 34, A: 255}
	case theme.ColorNamePlaceHolder:
		return placeH
	case theme.ColorNamePressed:
		return btnHover
	case theme.ColorNamePrimary:
		return accent
	case theme.ColorNameScrollBar:
		return scrollFg
	case theme.ColorNameScrollBarBackground:
		return scrollBg
	case theme.ColorNameSelection:
		return selBg
	case theme.ColorNameSeparator:
		return separator
	case theme.ColorNameShadow:
		return shadow
	case theme.ColorNameSuccess:
		return success
	case theme.ColorNameWarning:
		return warnYel
	}
	return fgLight
}

func (d *DarkMonospace) Font(_ fyne.TextStyle) fyne.Resource {
	return d.font
}

func (d *DarkMonospace) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

func (d *DarkMonospace) Size(name fyne.ThemeSizeName) float32 {
	switch name {
	case theme.SizeNameText:
		return 13
	case theme.SizeNameHeadingText:
		return 18
	case theme.SizeNameSubHeadingText:
		return 15
	case theme.SizeNameCaptionText:
		return 11
	case theme.SizeNameInlineIcon:
		return 14
	case theme.SizeNamePadding:
		return 4
	case theme.SizeNameInnerPadding:
		return 4
	case theme.SizeNameLineSpacing:
		return 2
	case theme.SizeNameScrollBar:
		return 6
	case theme.SizeNameScrollBarSmall:
		return 3
	case theme.SizeNameSeparatorThickness:
		return 1
	case theme.SizeNameSplitThickness:
		return 1
	case theme.SizeNameInnerWindowRadius:
		return 0
	case theme.SizeNameInputBorder:
		return 1
	case theme.SizeNameInputRadius:
		return 0
	case theme.SizeNameButtonRadius:
		return 0
	case theme.SizeNameCardRadius:
		return 0
	case theme.SizeNameDialogRadius:
		return 0
	case theme.SizeNamePopupRadius:
		return 0
	case theme.SizeNameMenuRadius:
		return 0
	case theme.SizeNameModalBlurRadius:
		return 0
	case theme.SizeNameSelectionRadius:
		return 0
	case theme.SizeNameScrollBarRadius:
		return 0
	case theme.SizeNameWindowButtonHeight:
		return 18
	case theme.SizeNameWindowButtonRadius:
		return 0
	case theme.SizeNameWindowButtonIcon:
		return 10
	case theme.SizeNameWindowTitleBarHeight:
		return 28
	}
	return 14
}
