import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ZoneTable } from './ZoneTable';
import { FieldItem, ZoneCollection } from '../types';

const mockField: FieldItem = {
  id: 'fld_test_punjab',
  name: 'Punjab Test Farm',
  crop_type: 'Wheat',
  total_acres: 500,
  center: [75.72, 30.85],
  boundary: [[[75.70, 30.84], [75.74, 30.84], [75.74, 30.86], [75.70, 30.86], [75.70, 30.84]]],
};

const mockZones: ZoneCollection = {
  type: 'FeatureCollection',
  job_id: 'job_test_123',
  scene_id: 'scn_test_123',
  generated_at: '2026-10-01T12:00:00Z',
  features: [
    {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[[75.721, 30.851], [75.725, 30.851], [75.725, 30.855], [75.721, 30.855], [75.721, 30.851]]],
      },
      properties: {
        zone_id: 'zn-urgent-high',
        risk_class: 2,
        risk_class_name: 'high_blight_risk',
        risk_score: 0.92,
        area_acres: 6.4,
        days_to_onset: 3.5, // Urgent (<= 7 days)
        dominant_indicator: 'red_edge_shift',
        recommended_action: 'Emergency fungicide spray within 48h.',
      },
    },
    {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[[75.715, 30.848], [75.719, 30.848], [75.719, 30.852], [75.715, 30.852], [75.715, 30.848]]],
      },
      properties: {
        zone_id: 'zn-early-stress',
        risk_class: 1,
        risk_class_name: 'early_stress',
        risk_score: 0.65,
        area_acres: 14.2,
        days_to_onset: 22.0,
        dominant_indicator: 'pri_decline',
        recommended_action: 'Scout canopy and deploy bio-stimulant.',
      },
    },
    {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[[75.712, 30.854], [75.718, 30.854], [75.718, 30.858], [75.712, 30.858], [75.712, 30.854]]],
      },
      properties: {
        zone_id: 'zn-healthy-vigor',
        risk_class: 0,
        risk_class_name: 'healthy',
        risk_score: 0.05,
        area_acres: 45.0,
        days_to_onset: 30.0,
        dominant_indicator: 'baseline',
        recommended_action: 'Maintain standard irrigation.',
      },
    },
  ],
};

describe('ZoneTable Component', () => {
  it('renders zone matrix header, distribution bar, and all matching zones for active field', () => {
    const handleSelect = vi.fn();
    render(
      <ZoneTable
        zones={mockZones}
        activeField={mockField}
        selectedZone={null}
        onSelectZone={handleSelect}
      />
    );

    expect(screen.getByText('Zone Risk Matrix')).toBeInTheDocument();
    expect(screen.getByText('3 of 3 zones')).toBeInTheDocument();
    expect(screen.getByText(/1 Critical Urgent/i)).toBeInTheDocument();
    expect(screen.getByText('Canopy Risk Distribution')).toBeInTheDocument();

    expect(screen.getByText('zn-urgent-high')).toBeInTheDocument();
    expect(screen.getByText('zn-early-stress')).toBeInTheDocument();
    expect(screen.getByText('zn-healthy-vigor')).toBeInTheDocument();
  });

  it('filters rows by free-text search input', () => {
    const handleSelect = vi.fn();
    render(
      <ZoneTable
        zones={mockZones}
        activeField={mockField}
        selectedZone={null}
        onSelectZone={handleSelect}
      />
    );

    const searchInput = screen.getByPlaceholderText(/Search by Zone ID/i);
    fireEvent.change(searchInput, { target: { value: 'urgent' } });

    expect(screen.getByText('zn-urgent-high')).toBeInTheDocument();
    expect(screen.queryByText('zn-early-stress')).not.toBeInTheDocument();
    expect(screen.queryByText('zn-healthy-vigor')).not.toBeInTheDocument();
  });

  it('filters rows by risk pill (urgent, early_stress, healthy)', () => {
    const handleSelect = vi.fn();
    render(
      <ZoneTable
        zones={mockZones}
        activeField={mockField}
        selectedZone={null}
        onSelectZone={handleSelect}
      />
    );

    // Click Urgent filter pill
    const urgentButton = screen.getByRole('button', { name: /Urgent/i });
    fireEvent.click(urgentButton);

    expect(screen.getByText('zn-urgent-high')).toBeInTheDocument();
    expect(screen.queryByText('zn-early-stress')).not.toBeInTheDocument();
    expect(screen.queryByText('zn-healthy-vigor')).not.toBeInTheDocument();

    // Click Early Stress filter pill
    const earlyButton = screen.getByRole('button', { name: /Early Stress/i });
    fireEvent.click(earlyButton);

    expect(screen.queryByText('zn-urgent-high')).not.toBeInTheDocument();
    expect(screen.getByText('zn-early-stress')).toBeInTheDocument();
    expect(screen.queryByText('zn-healthy-vigor')).not.toBeInTheDocument();

    // Click Healthy filter pill
    const healthyButton = screen.getByRole('button', { name: /Healthy/i });
    fireEvent.click(healthyButton);

    expect(screen.getByText('zn-healthy-vigor')).toBeInTheDocument();
    expect(screen.queryByText('zn-urgent-high')).not.toBeInTheDocument();
  });

  it('calls onSelectZone when a row is clicked', () => {
    const handleSelect = vi.fn();
    render(
      <ZoneTable
        zones={mockZones}
        activeField={mockField}
        selectedZone={null}
        onSelectZone={handleSelect}
      />
    );

    const row = screen.getByTestId('zone-row-zn-urgent-high');
    fireEvent.click(row);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleSelect).toHaveBeenCalledWith(mockZones.features[0]);
  });

  it('calls onInspectZone and onSelectZone when clicking inspect button', () => {
    const handleSelect = vi.fn();
    const handleInspect = vi.fn();
    render(
      <ZoneTable
        zones={mockZones}
        activeField={mockField}
        selectedZone={null}
        onSelectZone={handleSelect}
        onInspectZone={handleInspect}
      />
    );

    const inspectBtn = screen.getByRole('button', { name: /Inspect zn-urgent-high/i });
    fireEvent.click(inspectBtn);

    expect(handleSelect).toHaveBeenCalledTimes(1);
    expect(handleInspect).toHaveBeenCalledTimes(1);
    expect(handleInspect).toHaveBeenCalledWith(mockZones.features[0]);
  });

  it('renders empty state when search finds no match and resets filters', () => {
    const handleSelect = vi.fn();
    render(
      <ZoneTable
        zones={mockZones}
        activeField={mockField}
        selectedZone={null}
        onSelectZone={handleSelect}
      />
    );

    const searchInput = screen.getByPlaceholderText(/Search by Zone ID/i);
    fireEvent.change(searchInput, { target: { value: 'nonexistent-zone-xyz' } });

    expect(screen.getByText('No matching zones found')).toBeInTheDocument();

    const resetBtn = screen.getByRole('button', { name: /Reset Filters/i });
    fireEvent.click(resetBtn);

    expect(screen.getByText('zn-urgent-high')).toBeInTheDocument();
    expect(screen.getByText('zn-early-stress')).toBeInTheDocument();
  });
});
