import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ExportModal } from './ExportModal';
import { ZoneCollection } from '../types';

const mockZones: ZoneCollection = {
  type: 'FeatureCollection',
  job_id: 'job_test_export_99',
  scene_id: 'scn_test_spectral_01',
  generated_at: '2026-09-30T10:00:00Z',
  features: [
    {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [75.8, 30.9],
            [75.81, 30.9],
            [75.81, 30.91],
            [75.8, 30.91],
            [75.8, 30.9],
          ],
        ],
      },
      properties: {
        zone_id: 'zn_early_01',
        risk_class: 1,
        risk_class_name: 'early_stress',
        risk_score: 0.82,
        area_acres: 24.5,
        days_to_onset: 18.0,
        dominant_indicator: 'pri_decline',
        recommended_action: 'Apply targeted bio-fungicide within 5 days.',
      },
    },
    {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [
          [
            [75.82, 30.92],
            [75.83, 30.92],
            [75.83, 30.93],
            [75.82, 30.93],
            [75.82, 30.92],
          ],
        ],
      },
      properties: {
        zone_id: 'zn_high_02',
        risk_class: 2,
        risk_class_name: 'high_blight_risk',
        risk_score: 0.95,
        area_acres: 8.4,
        days_to_onset: 4.2,
        dominant_indicator: 'red_edge_shift',
        recommended_action: 'Immediate systemic strobilurin barrier treatment.',
      },
    },
  ],
};

describe('ExportModal Component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders nothing when isOpen is false', () => {
    const handleClose = vi.fn();
    const { container } = render(
      <ExportModal
        isOpen={false}
        onClose={handleClose}
        zones={mockZones}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('renders standard export menu options when isOpen is true', () => {
    const handleClose = vi.fn();
    render(
      <ExportModal
        isOpen={true}
        onClose={handleClose}
        zones={mockZones}
        fieldName="Punjab Wheat Parcel B"
      />
    );

    expect(screen.getByText('Export Agronomic Forecast')).toBeInTheDocument();
    expect(screen.getByText('Job: job_test_export_99')).toBeInTheDocument();
    expect(
      screen.getByText(/Agronomic Spray-Plan & Prescription \(PDF \/ Print\)/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Contract C4 GeoJSON Polygons/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Variable-Rate Spray Plan \(\.csv\)/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Copy GeoJSON Payload/i)
    ).toBeInTheDocument();
  });

  it('triggers GeoJSON download with valid blob and file name', () => {
    const handleClose = vi.fn();
    const createObjectURLMock = vi.fn().mockReturnValue('blob:http://localhost/geojson-123');
    const revokeObjectURLMock = vi.fn();
    globalThis.URL.createObjectURL = createObjectURLMock;
    globalThis.URL.revokeObjectURL = revokeObjectURLMock;

    render(
      <ExportModal
        isOpen={true}
        onClose={handleClose}
        zones={mockZones}
      />
    );

    const geoJsonOption = screen.getByText(/Contract C4 GeoJSON Polygons/i);
    fireEvent.click(geoJsonOption);

    expect(createObjectURLMock).toHaveBeenCalled();
    expect(revokeObjectURLMock).toHaveBeenCalled();
  });

  it('triggers CSV spray plan download with valid blob and file name', () => {
    const handleClose = vi.fn();
    const createObjectURLMock = vi.fn().mockReturnValue('blob:http://localhost/csv-123');
    const revokeObjectURLMock = vi.fn();
    globalThis.URL.createObjectURL = createObjectURLMock;
    globalThis.URL.revokeObjectURL = revokeObjectURLMock;

    render(
      <ExportModal
        isOpen={true}
        onClose={handleClose}
        zones={mockZones}
      />
    );

    const csvOption = screen.getByText(/Variable-Rate Spray Plan \(\.csv\)/i);
    fireEvent.click(csvOption);

    expect(createObjectURLMock).toHaveBeenCalled();
    expect(revokeObjectURLMock).toHaveBeenCalled();
  });

  it('copies GeoJSON payload to clipboard when copy option clicked', () => {
    const handleClose = vi.fn();
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, {
      clipboard: {
        writeText: writeTextMock,
      },
    });

    render(
      <ExportModal
        isOpen={true}
        onClose={handleClose}
        zones={mockZones}
      />
    );

    const copyOption = screen.getByText('Copy GeoJSON Payload');
    fireEvent.click(copyOption);

    expect(writeTextMock).toHaveBeenCalledWith(
      JSON.stringify(mockZones, null, 2)
    );
  });

  it('transitions to full printable report view when clicking PDF option', () => {
    const handleClose = vi.fn();
    const printMock = vi.fn();
    window.print = printMock;

    render(
      <ExportModal
        isOpen={true}
        onClose={handleClose}
        zones={mockZones}
        fieldName="Punjab Experimental Agronomy Field"
        cropType="Durum Wheat (HD-2967)"
        totalAcres={850}
      />
    );

    const pdfOption = screen.getByText(/Agronomic Spray-Plan & Prescription \(PDF \/ Print\)/i);
    fireEvent.click(pdfOption);

    // Assert document title & metadata rendered
    expect(
      screen.getByText('VARIABLE-RATE SPRAY PLAN & AGRONOMIC PRESCRIPTION')
    ).toBeInTheDocument();
    expect(screen.getByText('Punjab Experimental Agronomy Field')).toBeInTheDocument();
    expect(screen.getByText('Durum Wheat (HD-2967)')).toBeInTheDocument();
    expect(screen.getByText('850 Acres')).toBeInTheDocument();

    // Assert Executive Scorecard
    expect(screen.getByText('32.9 ac')).toBeInTheDocument(); // 24.5 + 8.4
    expect(screen.getByText('~4.2 Days')).toBeInTheDocument(); // Earliest onset

    // Assert Tabular Prescription
    expect(screen.getByText('zn_early_01')).toBeInTheDocument();
    expect(screen.getByText('zn_high_02')).toBeInTheDocument();
    expect(screen.getByText(/Bacillus subtilis QST 713/i)).toBeInTheDocument();
    expect(screen.getByText(/Azoxystrobin 250 SC/i)).toBeInTheDocument();

    // Assert Drone & Environmental specifications
    expect(screen.getByText(/UAV Flight Parameters/i)).toBeInTheDocument();
    expect(screen.getByText(/Environmental Application Limits/i)).toBeInTheDocument();

    // Test print action
    const printButton = screen.getByRole('button', { name: /Print \/ Save as PDF/i });
    fireEvent.click(printButton);
    expect(printMock).toHaveBeenCalledTimes(1);

    // Test return to menu
    const backButton = screen.getByRole('button', { name: /Back to Export Options/i });
    fireEvent.click(backButton);
    expect(screen.getByText('Export Agronomic Forecast')).toBeInTheDocument();
  });

  it('invokes onClose when close button is clicked', () => {
    const handleClose = vi.fn();
    render(
      <ExportModal
        isOpen={true}
        onClose={handleClose}
        zones={mockZones}
      />
    );

    const closeButton = screen.getByLabelText('Close export modal');
    fireEvent.click(closeButton);

    expect(handleClose).toHaveBeenCalledTimes(1);
  });
});
