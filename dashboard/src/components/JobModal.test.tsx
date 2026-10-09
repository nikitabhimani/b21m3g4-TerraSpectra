import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { JobModal } from './JobModal';
import { ApiService } from '../services/api';
import { SAMPLE_FIELDS, SAMPLE_SCENES } from '../fixtures/mockData';

describe('JobModal Component', () => {
  beforeEach(() => {
    ApiService.setDemoMode(true);
    vi.restoreAllMocks();
  });

  afterEach(() => {
    ApiService.setDemoMode(false);
    vi.restoreAllMocks();
  });

  it('renders correctly when open', () => {
    const handleClose = vi.fn();
    const handleComplete = vi.fn();

    render(
      <JobModal
        isOpen={true}
        onClose={handleClose}
        scenes={SAMPLE_SCENES}
        fields={SAMPLE_FIELDS}
        onJobComplete={handleComplete}
      />
    );

    expect(screen.getByText('Launch Hyperspectral Pre-Visual Scan')).toBeInTheDocument();
    expect(screen.getByText('Run Inference')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
  });

  it('does not render when isOpen is false', () => {
    render(
      <JobModal
        isOpen={false}
        onClose={vi.fn()}
        scenes={SAMPLE_SCENES}
        fields={SAMPLE_FIELDS}
        onJobComplete={vi.fn()}
      />
    );

    expect(screen.queryByText('Launch Hyperspectral Pre-Visual Scan')).not.toBeInTheDocument();
  });

  it('calls onClose when clicking Cancel button', () => {
    const handleClose = vi.fn();
    render(
      <JobModal
        isOpen={true}
        onClose={handleClose}
        scenes={SAMPLE_SCENES}
        fields={SAMPLE_FIELDS}
        onJobComplete={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Cancel'));
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('starts inference scan and reveals Abort Scan button', async () => {
    const createJobSpy = vi.spyOn(ApiService, 'createJob').mockResolvedValue({
      job_id: 'job_test_101',
      scene_id: SAMPLE_SCENES[0].scene_id,
      field_id: SAMPLE_FIELDS[0].id,
      status: 'queued',
      progress: 0.1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    render(
      <JobModal
        isOpen={true}
        onClose={vi.fn()}
        scenes={SAMPLE_SCENES}
        fields={SAMPLE_FIELDS}
        onJobComplete={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Run Inference'));

    await waitFor(() => {
      expect(createJobSpy).toHaveBeenCalled();
    });

    expect(screen.getByText('Abort Scan')).toBeInTheDocument();
  });

  it('aborts scan when Abort Scan button is clicked', async () => {
    vi.spyOn(ApiService, 'createJob').mockResolvedValue({
      job_id: 'job_test_cancel_ui',
      scene_id: SAMPLE_SCENES[0].scene_id,
      field_id: SAMPLE_FIELDS[0].id,
      status: 'running',
      progress: 0.2,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    const cancelJobSpy = vi.spyOn(ApiService, 'cancelJob').mockResolvedValue({
      job_id: 'job_test_cancel_ui',
      scene_id: SAMPLE_SCENES[0].scene_id,
      status: 'cancelled',
      progress: 0.2,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      error: 'Cancelled by user',
    });

    render(
      <JobModal
        isOpen={true}
        onClose={vi.fn()}
        scenes={SAMPLE_SCENES}
        fields={SAMPLE_FIELDS}
        onJobComplete={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Run Inference'));

    await waitFor(() => {
      expect(screen.getByText('Abort Scan')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Abort Scan'));

    await waitFor(() => {
      expect(cancelJobSpy).toHaveBeenCalledWith('job_test_cancel_ui');
      expect(screen.getByText(/Inference scan was cancelled/i)).toBeInTheDocument();
    });

    expect(screen.getByText('Retry Inference')).toBeInTheDocument();
  });

  it('allows retrying a cancelled scan', async () => {
    vi.spyOn(ApiService, 'createJob').mockResolvedValue({
      job_id: 'job_test_retry_ui',
      scene_id: SAMPLE_SCENES[0].scene_id,
      field_id: SAMPLE_FIELDS[0].id,
      status: 'running',
      progress: 0.2,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    vi.spyOn(ApiService, 'cancelJob').mockResolvedValue({
      job_id: 'job_test_retry_ui',
      scene_id: SAMPLE_SCENES[0].scene_id,
      status: 'cancelled',
      progress: 0.2,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      error: 'Cancelled by user',
    });

    const retryJobSpy = vi.spyOn(ApiService, 'retryJob').mockResolvedValue({
      job_id: 'job_test_retry_ui',
      scene_id: SAMPLE_SCENES[0].scene_id,
      status: 'queued',
      progress: 0.05,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    render(
      <JobModal
        isOpen={true}
        onClose={vi.fn()}
        scenes={SAMPLE_SCENES}
        fields={SAMPLE_FIELDS}
        onJobComplete={vi.fn()}
      />
    );

    fireEvent.click(screen.getByText('Run Inference'));

    await waitFor(() => {
      expect(screen.getByText('Abort Scan')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Abort Scan'));

    await waitFor(() => {
      expect(screen.getByText('Retry Inference')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Retry Inference'));

    await waitFor(() => {
      expect(retryJobSpy).toHaveBeenCalledWith('job_test_retry_ui');
    });
  });
});
