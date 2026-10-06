import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { Header } from './Header';

describe('Header Component', () => {
  const defaultProps = {
    apiStatus: 'ok',
    isDemoMode: false,
    onToggleDemoMode: vi.fn(),
    onOpenJobModal: vi.fn(),
    onOpenExportModal: vi.fn(),
    activeJobId: 'job_c4_live_rehearsal_001',
  };

  it('renders branding title and subtitle', () => {
    render(<Header {...defaultProps} />);

    expect(screen.getByText('TerraSpectra')).toBeInTheDocument();
    expect(screen.getByText('GIS 3D')).toBeInTheDocument();
    expect(
      screen.getByText(/Hyperspectral Pre-Visual Crop Disease Forecast/i)
    ).toBeInTheDocument();
  });

  it('renders live API status badge when online and demo mode is false', () => {
    render(<Header {...defaultProps} apiStatus="ok" isDemoMode={false} />);

    expect(screen.getByText(/API Live \(Port 8000\)/i)).toBeInTheDocument();
    expect(screen.queryByText(/Demo Fixture Active/i)).not.toBeInTheDocument();
    expect(screen.getByText('Live API Mode')).toBeInTheDocument();
  });

  it('renders demo fixture active badge when in demo mode', () => {
    render(<Header {...defaultProps} apiStatus="ok" isDemoMode={true} />);

    expect(screen.getByText(/Demo Fixture Active/i)).toBeInTheDocument();
    expect(screen.queryByText(/API Live \(Port 8000\)/i)).not.toBeInTheDocument();
    expect(screen.getByText('Demo Mode')).toBeInTheDocument();
  });

  it('invokes onToggleDemoMode when clicking demo mode button', () => {
    const handleToggle = vi.fn();
    render(<Header {...defaultProps} onToggleDemoMode={handleToggle} isDemoMode={true} />);

    const toggleBtn = screen.getByRole('button', { name: /Demo Mode/i });
    fireEvent.click(toggleBtn);

    expect(handleToggle).toHaveBeenCalledTimes(1);
  });

  it('invokes onOpenJobModal when clicking Run Pre-Visual Scan button', () => {
    const handleJobModal = vi.fn();
    render(<Header {...defaultProps} onOpenJobModal={handleJobModal} />);

    const scanBtn = screen.getByRole('button', { name: /Run Pre-Visual Scan/i });
    fireEvent.click(scanBtn);

    expect(handleJobModal).toHaveBeenCalledTimes(1);
  });

  it('invokes onOpenExportModal when clicking Export Plan button', () => {
    const handleExportModal = vi.fn();
    render(<Header {...defaultProps} onOpenExportModal={handleExportModal} />);

    const exportBtn = screen.getByRole('button', { name: /Export Plan/i });
    fireEvent.click(exportBtn);

    expect(handleExportModal).toHaveBeenCalledTimes(1);
  });
});
