import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ForecastSlider } from './ForecastSlider';

describe('ForecastSlider Component', () => {
  it('renders pre-visual incubation stage for horizon > 7 days', () => {
    const handleChange = vi.fn();
    render(<ForecastSlider daysHorizon={21} onChangeDays={handleChange} />);

    expect(screen.getByText('Forecast Horizon Timeline')).toBeInTheDocument();
    expect(screen.getByText('T - 21 Days')).toBeInTheDocument();
    expect(screen.getByText('Pre-Visual Incubation')).toBeInTheDocument();
  });

  it('renders acute blight spread stage for horizon <= 7 days and > 0', () => {
    const handleChange = vi.fn();
    render(<ForecastSlider daysHorizon={5} onChangeDays={handleChange} />);

    expect(screen.getByText('T - 5 Days')).toBeInTheDocument();
    expect(screen.getByText('Acute Blight Spread')).toBeInTheDocument();
  });

  it('renders visible outbreak stage for horizon = 0 days', () => {
    const handleChange = vi.fn();
    render(<ForecastSlider daysHorizon={0} onChangeDays={handleChange} />);

    expect(screen.getByText('T - 0 Days')).toBeInTheDocument();
    expect(screen.getByText('Today (Visible Outbreak)')).toBeInTheDocument();
  });

  it('calls onChangeDays when slider input changes', () => {
    const handleChange = vi.fn();
    render(<ForecastSlider daysHorizon={15} onChangeDays={handleChange} />);

    const slider = screen.getByRole('slider');
    fireEvent.change(slider, { target: { value: '8' } });

    expect(handleChange).toHaveBeenCalledWith(8);
  });

  it('resets to 30 days when reset button is clicked', () => {
    const handleChange = vi.fn();
    render(<ForecastSlider daysHorizon={10} onChangeDays={handleChange} />);

    const resetButton = screen.getByTitle('Reset to 30 days prior');
    fireEvent.click(resetButton);

    expect(handleChange).toHaveBeenCalledWith(30);
  });
});
