import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { RoiCalculator } from './RoiCalculator';

describe('RoiCalculator Component', () => {
  it('renders economic ROI banner and title', () => {
    render(<RoiCalculator atRiskAcres={10} totalAcres={100} />);
    expect(screen.getByText('Agronomic Economic ROI & Savings')).toBeInTheDocument();
    expect(screen.getByText('3-Week Pre-Visual Window')).toBeInTheDocument();
  });

  it('correctly calculates crop value at risk and chemical savings', () => {
    // atRiskAcres = 10, totalAcres = 100
    // totalValueAtRisk = 10 * 1250 = $12,500
    // blanketChemicalCost = 100 * 14.5 = $1,450
    // targetedChemicalCost = 10 * 22.0 = $220
    // chemicalSavings = 1450 - 220 = $1,230
    // chemicalSavedPct = round((1230 / 1450) * 100) = 85%
    render(<RoiCalculator atRiskAcres={10} totalAcres={100} />);

    expect(screen.getByText('$12,500')).toBeInTheDocument();
    expect(screen.getByText('10.0 acres blight prevented')).toBeInTheDocument();
    expect(screen.getByText('85% Saved')).toBeInTheDocument();
    expect(screen.getByText('$1,230 input cost saved')).toBeInTheDocument();
    expect(screen.getByText('$220')).toBeInTheDocument();
    expect(screen.getByText('$1450')).toBeInTheDocument();
  });

  it('handles 0 at-risk acres gracefully without negative savings', () => {
    render(<RoiCalculator atRiskAcres={0} totalAcres={50} />);

    const zeroElements = screen.getAllByText('$0');
    expect(zeroElements.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('0.0 acres blight prevented')).toBeInTheDocument();
    // blanket = 50 * 14.5 = $725, targeted = 0
    // savings = $725 (100% saved)
    expect(screen.getByText('100% Saved')).toBeInTheDocument();
    expect(screen.getByText('$725 input cost saved')).toBeInTheDocument();
  });
});
