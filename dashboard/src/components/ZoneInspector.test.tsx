import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ZoneInspector } from './ZoneInspector';
import { ZoneFeature } from '../types';

const mockZoneEarlyStress: ZoneFeature = {
  type: 'Feature',
  geometry: {
    type: 'Polygon',
    coordinates: [[[75.8, 30.9], [75.81, 30.9], [75.81, 30.91], [75.8, 30.91], [75.8, 30.9]]],
  },
  properties: {
    zone_id: 'zn_test_01',
    risk_class: 1,
    risk_class_name: 'early_stress',
    risk_score: 0.88,
    area_acres: 14.5,
    days_to_onset: 21.0,
    dominant_indicator: 'pri_decline',
    recommended_action: 'Apply targeted bio-fungicide within 5 days.',
  },
};

const mockZoneHighRisk: ZoneFeature = {
  type: 'Feature',
  geometry: {
    type: 'Polygon',
    coordinates: [[[75.8, 30.9], [75.81, 30.9], [75.81, 30.91], [75.8, 30.91], [75.8, 30.9]]],
  },
  properties: {
    zone_id: 'zn_test_02',
    risk_class: 2,
    risk_class_name: 'high_blight_risk',
    risk_score: 0.94,
    area_acres: 5.2,
    days_to_onset: 4.5,
    dominant_indicator: 'red_edge_shift',
    recommended_action: 'Emergency systemic fungicide spray required.',
  },
};

describe('ZoneInspector Component', () => {
  it('renders empty state placeholder when no zone is selected', () => {
    const handleClear = vi.fn();
    render(<ZoneInspector selectedZone={null} onClearSelection={handleClear} />);

    expect(screen.getByText('No Zone Selected')).toBeInTheDocument();
    expect(
      screen.getByText(/Click any polygon zone or heatmap anomaly/i)
    ).toBeInTheDocument();
  });

  it('renders zone details, indicators, and actions for early stress zone', () => {
    const handleClear = vi.fn();
    render(<ZoneInspector selectedZone={mockZoneEarlyStress} onClearSelection={handleClear} />);

    expect(screen.getByText('zn_test_01')).toBeInTheDocument();
    expect(screen.getByText('Pre-Visual Stress (Class 1)')).toBeInTheDocument();
    expect(screen.getByText('PRI Xanthophyll Cycle Decline (531–570 nm)')).toBeInTheDocument();
    expect(screen.getByText('88.0%')).toBeInTheDocument();
    expect(screen.getByText(/14.5 Acres/i)).toBeInTheDocument();
    expect(screen.getByText('21.0 Days')).toBeInTheDocument();
    expect(screen.getByText(/Apply targeted bio-fungicide within 5 days/i)).toBeInTheDocument();
  });

  it('renders red-edge shift indicator for high risk zone', () => {
    const handleClear = vi.fn();
    render(<ZoneInspector selectedZone={mockZoneHighRisk} onClearSelection={handleClear} />);

    expect(screen.getByText('High Blight Risk (Class 2)')).toBeInTheDocument();
    expect(screen.getByText('Red-Edge Blue Shift (705 nm)')).toBeInTheDocument();
    expect(screen.getByText('94.0%')).toBeInTheDocument();
    expect(screen.getByText(/5.2 Acres/i)).toBeInTheDocument();
    expect(screen.getByText('4.5 Days')).toBeInTheDocument();
  });

  it('invokes onClearSelection when clear button is clicked', () => {
    const handleClear = vi.fn();
    render(<ZoneInspector selectedZone={mockZoneEarlyStress} onClearSelection={handleClear} />);

    const clearButton = screen.getByRole('button', { name: /Clear/i });
    fireEvent.click(clearButton);

    expect(handleClear).toHaveBeenCalledTimes(1);
  });
});
