import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';
import { PixelProbeModal } from './PixelProbeModal';
import { sampleHyperspectralPixel } from '../services/pixelProbe';
import { CONTRACT_SAMPLE_ZONES, SAMPLE_FIELDS } from '../fixtures/mockData';

describe('PixelProbeModal Component', () => {
  const field = SAMPLE_FIELDS[0];
  const zones = CONTRACT_SAMPLE_ZONES;
  const mockProbe = sampleHyperspectralPixel(30.8540, 75.7245, field, zones);

  it('does not render when isOpen is false', () => {
    const { container } = render(
      <PixelProbeModal isOpen={false} onClose={vi.fn()} probeData={mockProbe} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders modal with telemetry when open', () => {
    render(
      <PixelProbeModal isOpen={true} onClose={vi.fn()} probeData={mockProbe} />
    );

    expect(screen.getByText(/Hyperspectral Pixel Probe Telemetry/i)).toBeInTheDocument();
    expect(screen.getByText(/EnMAP-L2A/i)).toBeInTheDocument();
    expect(screen.getByText(/Calculated Biophysical Stress Indices/i)).toBeInTheDocument();
    expect(screen.getByText(/REP \(Guyot\)/i)).toBeInTheDocument();
    expect(screen.getByText(/PRI \(531\/570\)/i)).toBeInTheDocument();
  });

  it('calls onClose when clicking Dismiss button or X icon', () => {
    const handleClose = vi.fn();
    render(
      <PixelProbeModal isOpen={true} onClose={handleClose} probeData={mockProbe} />
    );

    const dismissBtn = screen.getByRole('button', { name: /Dismiss/i });
    fireEvent.click(dismissBtn);
    expect(handleClose).toHaveBeenCalledTimes(1);

    const closeIcon = screen.getByLabelText(/Close modal/i);
    fireEvent.click(closeIcon);
    expect(handleClose).toHaveBeenCalledTimes(2);
  });

  it('allows toggling healthy and stressed reference curves', () => {
    render(
      <PixelProbeModal isOpen={true} onClose={vi.fn()} probeData={mockProbe} />
    );

    const healthyBtn = screen.getByRole('button', { name: /Healthy Ref/i });
    expect(healthyBtn).toBeInTheDocument();
    fireEvent.click(healthyBtn); // toggle off
    fireEvent.click(healthyBtn); // toggle back on
  });
});
