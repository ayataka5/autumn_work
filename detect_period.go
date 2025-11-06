package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/xuri/excelize/v2"
)

var debugGlobal = false

// nextPow2 returns smallest power of two >= n
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// fftComplex performs an in-place Cooley-Tukey radix-2 FFT and returns a new slice.
// If inverse is true, it computes the inverse transform (unnormalized).
func fftComplex(input []complex128, inverse bool) []complex128 {
	n := len(input)
	if n == 0 {
		return nil
	}
	// copy input
	a := make([]complex128, n)
	copy(a, input)

	// bit-reversal permutation
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}

	// choose sign convention: forward uses exp(-2πi/len)
	for length := 2; length <= n; length <<= 1 {
		ang := -2.0 * math.Pi / float64(length)
		if inverse {
			ang = -ang
		}
		wlen := complex(math.Cos(ang), math.Sin(ang))
		for i := 0; i < n; i += length {
			w := complex(1.0, 0.0)
			half := length >> 1
			for k := 0; k < half; k++ {
				u := a[i+k]
				v := a[i+k+half] * w
				a[i+k] = u + v
				a[i+k+half] = u - v
				w *= wlen
			}
		}
	}
	return a
}

// FFTReal converts real input to complex, runs FFT and returns complex spectrum.
func FFTReal(x []float64) []complex128 {
	n := len(x)
	cx := make([]complex128, n)
	for i := 0; i < n; i++ {
		cx[i] = complex(x[i], 0)
	}
	return fftComplex(cx, false)
}

// IFFT computes the inverse FFT of complex input and returns complex time-domain (normalized).
func IFFT(X []complex128) []complex128 {
	y := fftComplex(X, true)
	n := complex(float64(len(y)), 0)
	for i := range y {
		y[i] /= n
	}
	return y
}

// downsampleAverage simple block-average decimation by factor d
func downsampleAverage(x []float64, d int) []float64 {
	if d <= 1 {
		out := make([]float64, len(x))
		copy(out, x)
		return out
	}
	m := len(x) / d
	out := make([]float64, m)
	for i := 0; i < m; i++ {
		sum := 0.0
		for j := 0; j < d; j++ {
			sum += x[i*d+j]
		}
		out[i] = sum / float64(d)
	}
	return out
}

// mean subtract mean
func subtractMean(x []float64) {
	sum := 0.0
	for _, v := range x {
		sum += v
	}
	mean := sum / float64(len(x))
	for i := range x {
		x[i] -= mean
	}
}

// autocorrFFT computes autocorrelation (lags 0..n-1) using FFT
func autocorrFFT(x []float64) []float64 {
	n := len(x)
	L := nextPow2(2 * n)
	// zero-pad to L
	padded := make([]float64, L)
	copy(padded, x)
	// FFTReal returns []complex128
	X := FFTReal(padded)
	// power = X * conj(X)
	for i := range X {
		X[i] = X[i] * complex(real(X[i]), -imag(X[i]))
	}
	// IFFT (complex) -> time domain (complex)
	ifft := IFFT(X)
	// real part contains circular autocorrelation; take first n lags
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = real(ifft[i])
	}
	// normalize by number of points (optional)
	// here normalize by autocorr[0] so the 0-lag = 1
	if out[0] != 0 {
		scl := out[0]
		for i := range out {
			out[i] /= scl
		}
	}
	return out
}

// crossCorrFFT computes linear cross-correlation r_xy[k] for k=-(m-1)..(n-1)
// returned slice has length n+m-1, index i corresponds to lag = i-(m-1)
func crossCorrFFT(x, y []float64) []float64 {
	n := len(x)
	m := len(y)
	if n == 0 || m == 0 {
		return nil
	}
	L := nextPow2(n + m - 1)
	px := make([]float64, L)
	py := make([]float64, L)
	copy(px, x)
	copy(py, y)
	X := FFTReal(px)
	Y := FFTReal(py)
	// multiply X * conj(Y)
	for i := range X {
		X[i] = X[i] * complex(real(Y[i]), -imag(Y[i]))
	}
	ifft := IFFT(X)
	outLen := n + m - 1
	out := make([]float64, outLen)
	for i := 0; i < outLen; i++ {
		out[i] = real(ifft[i])
	}
	return out
}

// findPeakParabola finds argmax in ac between minIdx and maxIdx (inclusive) and fits a parabola
// around the integer peak to return sub-index and value.
func findPeakParabola(ac []float64, minIdx, maxIdx int) (float64, float64) {
	if minIdx < 0 {
		minIdx = 0
	}
	if maxIdx >= len(ac) {
		maxIdx = len(ac) - 1
	}
	best := minIdx
	bestVal := ac[minIdx]
	for i := minIdx + 1; i <= maxIdx; i++ {
		if ac[i] > bestVal {
			best = i
			bestVal = ac[i]
		}
	}
	if best <= 0 || best >= len(ac)-1 {
		return float64(best), ac[best]
	}
	y1 := ac[best-1]
	y2 := ac[best]
	y3 := ac[best+1]
	den := (y1 - 2*y2 + y3)
	var dx float64
	if den == 0 {
		dx = 0
	} else {
		dx = 0.5 * (y1 - y3) / den
		if math.Abs(dx) > 0.9 {
			dx = 0
		}
	}
	peakIdx := float64(best) + dx
	peakVal := y2 - 0.25*(y1-y3)*dx
	return peakIdx, peakVal
}

// findPeakWithParabola finds argmax in [minLag,maxLag], and does parabolic interpolation for sub-sample peak
func findPeakWithParabola(ac []float64, minLag, maxLag int) (peakLag float64, peakVal float64) {
	if minLag < 1 {
		minLag = 1
	}
	if maxLag >= len(ac)-1 {
		maxLag = len(ac) - 2
	}
	// find integer max
	best := minLag
	bestVal := ac[minLag]
	for i := minLag + 1; i <= maxLag; i++ {
		if ac[i] > bestVal {
			best = i
			bestVal = ac[i]
		}
	}
	// parabolic interpolation around best (use best-1, best, best+1)
	y1 := ac[best-1]
	y2 := ac[best]
	y3 := ac[best+1]
	// vertex offset from center = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
	den := (y1 - 2*y2 + y3)
	var dx float64
	if den == 0 {
		dx = 0
	} else {
		dx = 0.5 * (y1 - y3) / den
		// clamp extreme extrapolations — if interpolation goes far outside neighbors it's unreliable
		if math.Abs(dx) > 0.9 {
			dx = 0
		}
	}
	peakLag = float64(best) + dx
	// ensure peakLag stays within valid interior range
	if peakLag < 1 {
		peakLag = 1
	}
	// estimate peak value from parabola
	peakVal = y2 - 0.25*(y1-y3)*dx
	return
}

func detectPeriodSamples(x []float64, approx int, searchPct float64, dsMax int, debug bool) (lagSamples float64, err error) {
	if len(x) < 10 {
		return 0, fmt.Errorf("too short input")
	}
	// subtract mean
	subtractMean(x)

	// decide downsample factor d: keep reduced period between 50..500
	d := 1
	if approx > 0 {
		target := 200 // target reduced-period
		d = int(math.Max(1, math.Floor(float64(approx)/float64(target))))
		if d > dsMax {
			d = dsMax
		}
	}
	// do downsample
	xd := downsampleAverage(x, d)

	// compute autocorr
	ac := autocorrFFT(xd)
	if debug {
		fmt.Printf("DEBUG: input len=%d, downsampled len=%d, d=%d, ac len=%d\n", len(x), len(xd), d, len(ac))
		// print first few input samples
		end := 10
		if len(x) < end {
			end = len(x)
		}
		fmt.Print("DEBUG: x[0..]:")
		for i := 0; i < end; i++ {
			fmt.Printf(" %g", x[i])
		}
		fmt.Println()
		// print first autocorr values
		acend := 40
		if len(ac) < acend {
			acend = len(ac)
		}
		fmt.Print("DEBUG: ac[0..]:")
		for i := 0; i < acend; i++ {
			fmt.Printf(" %g", ac[i])
		}
		fmt.Println()
		// compute direct autocorr for first few lags to verify FFT result
		maxLagCheck := 40
		if maxLagCheck > len(xd)-1 {
			maxLagCheck = len(xd) - 1
		}
		fmt.Print("DEBUG: direct ac[0..]:")
		for lag := 0; lag <= maxLagCheck; lag++ {
			s := 0.0
			for i := 0; i+lag < len(xd); i++ {
				s += xd[i] * xd[i+lag]
			}
			fmt.Printf(" %g", s)
		}
		fmt.Println()
	}

	// compute search window in downsampled domain
	minLag := int(math.Max(1.0, float64(approx)*(1.0-searchPct)/float64(d)))
	maxLag := int(math.Min(float64(len(ac)-2), float64(approx)*(1.0+searchPct)/float64(d)))

	// in case approx unknown, fallback to reasonable range
	if approx == 0 {
		minLag = 1
		maxLag = int(math.Min(float64(len(ac)-2), 0.8*float64(len(ac))))
	}

	// for debug also find integer best
	if debug {
		best := minLag
		bestVal := ac[minLag]
		for i := minLag + 1; i <= maxLag; i++ {
			if ac[i] > bestVal {
				best = i
				bestVal = ac[i]
			}
		}
		fmt.Printf("DEBUG: search window minLag=%d maxLag=%d integer best=%d val=%g\n", minLag, maxLag, best, bestVal)
	}
	lag, _ := findPeakWithParabola(ac, minLag, maxLag)
	lagSamples = lag * float64(d) // convert back to original sample units
	return lagSamples, nil
}

func main() {
	// Command-line flags
	infile := flag.String("in", "", "input Excel .xlsx file path (required)")
	sheet := flag.String("sheet", "", "sheet name (optional, default first sheet)")
	col := flag.Int("col", 1, "1-based column index to read values from")
	Fs := flag.Float64("fs", 100.0, "sampling frequency (Hz)")
	approx := flag.Int("approx", 0, "approximate period in samples (optional)")
	searchPct := flag.Float64("searchPct", 0.2, "search window percentage (e.g. 0.2 for ±20%)")
	dsMax := flag.Int("dsMax", 20, "max downsample factor")
	debug := flag.Bool("debug", false, "print debug info about samples and autocorr")
	tRange := flag.String("tRange", "", "R1C1 range for time values, e.g. R17C8:R9999C8")
	x1Range := flag.String("x1Range", "", "R1C1 range for x1 values, e.g. R17C9:R9999C9")
	x2Range := flag.String("x2Range", "", "R1C1 range for x2 values, e.g. R17C11:R9999C11")
	doXcorr := flag.Bool("xcorr", false, "compute cross-correlation between x1 and x2 and print lag")
	flag.Parse()

	if *infile == "" {
		fmt.Fprintln(os.Stderr, "-in flag required: path to .xlsx file")
		flag.Usage()
		os.Exit(2)
	}

	// If any of the R1C1 ranges are provided, read those specific ranges.
	used := false
	var err error
	if *tRange != "" || *x1Range != "" || *x2Range != "" {
		used = true
		// read t if provided
		var tvals []float64
			if *tRange != "" {
				tvals, err = readRangeFloat(*infile, *sheet, *tRange)
			if err != nil {
				fmt.Fprintln(os.Stderr, "failed to read tRange:", err)
				os.Exit(1)
			}
		}

		// helper to compute and print for a given x range label
		handleX := func(label, r string) {
			if r == "" {
				return
			}
			xvals, err := readRangeFloat(*infile, *sheet, r)
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed to read %s: %v\n", label, err)
				os.Exit(1)
			}
			lag, err := detectPeriodSamples(xvals, *approx, *searchPct, *dsMax, *debug)
			if err != nil {
				fmt.Fprintf(os.Stderr, "error on %s: %v\n", label, err)
				os.Exit(1)
			}
			if len(tvals) > 1 {
				// compute mean dt from tvals
				sumdt := 0.0
				cnt := 0
				for i := 1; i < len(tvals) && i < len(xvals); i++ {
					dt := tvals[i] - tvals[i-1]
					if dt > 0 {
						sumdt += dt
						cnt++
					}
				}
				if cnt > 0 {
					meanDt := sumdt / float64(cnt)
					fmt.Printf("%s: estimated period (samples): %.3f, period seconds: %.6g s\n", label, lag, lag*meanDt)
					return
				}
			}
			// fallback to Fs flag
			fmt.Printf("%s: estimated period (samples): %.3f, period seconds (Fs=%.6g): %.6g s\n", label, lag, *Fs, lag/ *Fs)
		}

		// set global debug flag for range reader to print cell-level info
		if *debug {
			debugGlobal = true
		}
		handleX("x1", *x1Range)
		handleX("x2", *x2Range)

		// optionally compute cross-correlation if requested and both ranges present
		if *doXcorr {
			if *x1Range == "" || *x2Range == "" {
				fmt.Fprintln(os.Stderr, "-xcorr requires both -x1Range and -x2Range")
				os.Exit(2)
			}
			x1, err := readRangeFloat(*infile, *sheet, *x1Range)
			if err != nil {
				fmt.Fprintln(os.Stderr, "failed to read x1Range for xcorr:", err)
				os.Exit(1)
			}
			x2, err := readRangeFloat(*infile, *sheet, *x2Range)
			if err != nil {
				fmt.Fprintln(os.Stderr, "failed to read x2Range for xcorr:", err)
				os.Exit(1)
			}
			// subtract mean
			xc := make([]float64, len(x1))
			yc := make([]float64, len(x2))
			copy(xc, x1)
			copy(yc, x2)
			subtractMean(xc)
			subtractMean(yc)
			cc := crossCorrFFT(xc, yc)
			// find peak index
			idx, val := findPeakParabola(cc, 0, len(cc)-1)
			// convert index to lag: lag = idx - (len(y)-1)
			lagSamples := idx - float64(len(yc)-1)
			// also compute normalized cross-correlation
			Ex := 0.0
			Ey := 0.0
			for i := range xc {
				Ex += xc[i] * xc[i]
			}
			for i := range yc {
				Ey += yc[i] * yc[i]
			}
			normVal := val
			if Ex > 0 && Ey > 0 {
				normVal = val / math.Sqrt(Ex*Ey)
			}
			// Now also search around zero lag (center) to avoid spurious end peaks
			center := len(yc) - 1
			// choose search radius: if approx provided, use approx*1.5, else 5000 or half-length
			var radius int
			if *approx > 0 {
				radius = int(float64(*approx) * (1.0 + *searchPct))
			} else {
				radius = 5000
			}
			if radius > len(cc)/2 {
				radius = len(cc) / 2
			}
			minIdx := center - radius
			maxIdx := center + radius
			if minIdx < 0 {
				minIdx = 0
			}
			if maxIdx >= len(cc) {
				maxIdx = len(cc) - 1
			}
			cIdx, cVal := findPeakParabola(cc, minIdx, maxIdx)
			cLag := cIdx - float64(len(yc)-1)
			cNorm := cVal
			if Ex > 0 && Ey > 0 {
				cNorm = cVal / math.Sqrt(Ex*Ey)
			}
			// compute seconds using tRange if available
			if *tRange != "" {
				tvals, err := readRangeFloat(*infile, *sheet, *tRange)
				if err == nil && len(tvals) > 1 {
					sumdt := 0.0
					cnt := 0
					for i := 1; i < len(tvals); i++ {
						dt := tvals[i] - tvals[i-1]
						if dt > 0 {
							sumdt += dt
							cnt++
						}
					}
					if cnt > 0 {
						meanDt := sumdt / float64(cnt)
						fmt.Printf("xcorr: raw peak lag = %.3f samples (%.6g s), raw val=%.6g, raw norm=%.6g\n", lagSamples, lagSamples*meanDt, val, normVal)
						fmt.Printf("xcorr: constrained peak lag = %.3f samples (%.6g s), val=%.6g, norm=%.6g\n", cLag, cLag*meanDt, cVal, cNorm)
					} else {
						fmt.Printf("xcorr: raw peak lag = %.3f samples, raw val=%.6g, raw norm=%.6g\n", lagSamples, val, normVal)
						fmt.Printf("xcorr: constrained peak lag = %.3f samples, val=%.6g, norm=%.6g\n", cLag, cVal, cNorm)
					}
				} else {
					fmt.Printf("xcorr: raw peak lag = %.3f samples, raw val=%.6g, raw norm=%.6g\n", lagSamples, val, normVal)
					fmt.Printf("xcorr: constrained peak lag = %.3f samples, val=%.6g, norm=%.6g\n", cLag, cVal, cNorm)
				}
			} else {
				// fallback to Fs
				fmt.Printf("xcorr: raw peak lag = %.3f samples (Fs=%.6g -> %.6g s), raw val=%.6g, raw norm=%.6g\n", lagSamples, *Fs, lagSamples/ *Fs, val, normVal)
				fmt.Printf("xcorr: constrained peak lag = %.3f samples (Fs=%.6g -> %.6g s), val=%.6g, norm=%.6g\n", cLag, *Fs, cLag/ *Fs, cVal, cNorm)
			}
		}
		debugGlobal = false
	}

	if !used {
		x, err := readXLSX(*infile, *sheet, *col)
		if err != nil {
			fmt.Fprintln(os.Stderr, "failed to read xlsx:", err)
			os.Exit(1)
		}

	lag, err := detectPeriodSamples(x, *approx, *searchPct, *dsMax, *debug)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		fmt.Printf("estimated period (samples): %.3f\n", lag)
		fmt.Printf("period seconds (Fs=%.6g): %.6g s\n", *Fs, lag/ *Fs)
	}
}

// simple random noise
// readXLSX reads numeric values from the given .xlsx file.
// It loads the first sheet if sheet=="" and reads the 1-based column `col`.
func readXLSX(path, sheet string, col int) ([]float64, error) {
	if col < 1 {
		return nil, fmt.Errorf("col must be >= 1")
	}
	f, err := excelize.OpenFile(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	if sheet == "" {
		sheets := f.GetSheetList()
		if len(sheets) == 0 {
			return nil, fmt.Errorf("no sheets in workbook")
		}
		sheet = sheets[0]
	}

	rows, err := f.GetRows(sheet)
	if err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(rows))
	for _, r := range rows {
		if len(r) < col {
			continue
		}
		s := strings.TrimSpace(r[col-1])
		if s == "" {
			continue
		}
		// normalize comma decimal separators
		s = strings.ReplaceAll(s, ",", ".")
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			// ignore non-numeric cells
			continue
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no numeric values found in %s (sheet=%s col=%d)", path, sheet, col)
	}
	return out, nil
}

// parseR1C1Range parses a range like R17C8:R9999C8 and returns startRow,startCol,endRow,endCol
func parseR1C1Range(r string) (int, int, int, int, error) {
	parts := strings.Split(r, ":")
	if len(parts) != 2 {
		return 0, 0, 0, 0, fmt.Errorf("invalid range format: %s", r)
	}
	parse := func(s string) (int, int, error) {
		s = strings.TrimSpace(s)
		if !strings.HasPrefix(s, "R") {
			return 0, 0, fmt.Errorf("expected R..C.. format, got %s", s)
		}
		s = s[1:]
		idx := strings.IndexByte(s, 'C')
		if idx < 1 {
			return 0, 0, fmt.Errorf("expected R..C.. format, got %s", s)
		}
		rStr := s[:idx]
		cStr := s[idx+1:]
		ri, err := strconv.Atoi(rStr)
		if err != nil {
			return 0, 0, err
		}
		ci, err := strconv.Atoi(cStr)
		if err != nil {
			return 0, 0, err
		}
		return ri, ci, nil
	}
	r1, c1, err := parse(parts[0])
	if err != nil {
		return 0, 0, 0, 0, err
	}
	r2, c2, err := parse(parts[1])
	if err != nil {
		return 0, 0, 0, 0, err
	}
	// normalize order
	if r1 > r2 {
		r1, r2 = r2, r1
	}
	if c1 > c2 {
		c1, c2 = c2, c1
	}
	return r1, c1, r2, c2, nil
}

// readRangeFloat reads a single-column R1C1 range (e.g. R17C9:R9999C9) and returns numeric values per row.
func readRangeFloat(path, sheet, r string) ([]float64, error) {
	sr, sc, er, ec, err := parseR1C1Range(r)
	if err != nil {
		return nil, err
	}
	if sc != ec {
		return nil, fmt.Errorf("readRangeFloat: only single-column ranges supported (got %d..%d)", sc, ec)
	}
	f, err := excelize.OpenFile(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	if sheet == "" {
		sheets := f.GetSheetList()
		if len(sheets) == 0 {
			return nil, fmt.Errorf("no sheets in workbook")
		}
		sheet = sheets[0]
	}
	out := make([]float64, 0, er-sr+1)
	for row := sr; row <= er; row++ {
		cell, err := excelize.CoordinatesToCellName(sc, row)
		if err != nil {
			continue
		}
		s, err := f.GetCellValue(sheet, cell)
		if err != nil {
			continue
		}
		if debugGlobal && len(out) < 10 {
			fmt.Printf("DEBUG: cell %s raw='%s'\n", cell, s)
		}
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		s = strings.ReplaceAll(s, ",", ".")
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			continue
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no numeric values in range %s (sheet=%s)", r, sheet)
	}
	return out, nil
}