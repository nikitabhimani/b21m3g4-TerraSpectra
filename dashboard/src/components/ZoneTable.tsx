import React, { useMemo, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowUpDown,
  CheckCircle2,
  ChevronRight,
  Clock,
  Crosshair,
  Eye,
  Filter,
  Flame,
  Leaf,
  Layers,
  Search,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  X,
} from 'lucide-react';
import { FieldItem, RiskClass, ZoneCollection, ZoneFeature } from '../types';

export interface ZoneTableProps {
  zones: ZoneCollection;
  activeField: FieldItem;
  selectedZone: ZoneFeature | null;
  onSelectZone: (zone: ZoneFeature) => void;
  onInspectZone?: (zone: ZoneFeature) => void;
  daysHorizon?: number;
}

type SortField = 'days_to_onset' | 'risk_score' | 'area_acres' | 'zone_id';
type SortDirection = 'asc' | 'desc';
type FilterFilter = 'all' | 'urgent' | 'high_risk' | 'early_stress' | 'healthy';

export const ZoneTable: React.FC<ZoneTableProps> = ({
  zones,
  activeField,
  selectedZone,
  onSelectZone,
  onInspectZone,
  daysHorizon = 30,
}) => {
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [activeFilter, setActiveFilter] = useState<FilterFilter>('all');
  const [sortField, setSortField] = useState<SortField>('days_to_onset');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [filterByHorizon, setFilterByHorizon] = useState<boolean>(false);

  const allFeatures = zones.features || [];

  // Filter features to current active field parcel
  const fieldFeatures = useMemo(() => {
    return allFeatures.filter((f) => {
      const ring = f.geometry.coordinates[0];
      if (!ring || !ring[0]) return false;
      return (
        Math.abs(ring[0][0] - activeField.center[0]) < 0.25 &&
        Math.abs(ring[0][1] - activeField.center[1]) < 0.25
      );
    });
  }, [allFeatures, activeField]);

  // Aggregate stats for the field
  const stats = useMemo(() => {
    const total = fieldFeatures.length;
    let healthyAcres = 0;
    let earlyAcres = 0;
    let highBlightAcres = 0;
    let visibleAcres = 0;
    let urgentCount = 0;

    for (const f of fieldFeatures) {
      const p = f.properties;
      if (p.risk_class === 0) healthyAcres += p.area_acres;
      else if (p.risk_class === 1) earlyAcres += p.area_acres;
      else if (p.risk_class === 2) highBlightAcres += p.area_acres;
      else if (p.risk_class === 3) visibleAcres += p.area_acres;

      if (p.risk_class > 0 && p.days_to_onset <= 7) {
        urgentCount++;
      }
    }

    const totalAcres = healthyAcres + earlyAcres + highBlightAcres + visibleAcres || 1;
    return {
      total,
      totalAcres: Number(totalAcres.toFixed(1)),
      urgentCount,
      healthyPct: Math.round((healthyAcres / totalAcres) * 100),
      earlyPct: Math.round((earlyAcres / totalAcres) * 100),
      highBlightPct: Math.round((highBlightAcres / totalAcres) * 100),
      visiblePct: Math.round((visibleAcres / totalAcres) * 100),
    };
  }, [fieldFeatures]);

  // Filter & Search processing
  const processedFeatures = useMemo(() => {
    return fieldFeatures
      .filter((f) => {
        const p = f.properties;

        // Days horizon check
        if (filterByHorizon && p.risk_class > 0) {
          if (p.days_to_onset < 30 - daysHorizon) {
            return false;
          }
        }

        // Category filter
        if (activeFilter === 'urgent' && (p.risk_class === 0 || p.days_to_onset > 7)) {
          return false;
        }
        if (activeFilter === 'high_risk' && p.risk_class < 2) {
          return false;
        }
        if (activeFilter === 'early_stress' && p.risk_class !== 1) {
          return false;
        }
        if (activeFilter === 'healthy' && p.risk_class !== 0) {
          return false;
        }

        // Text search
        if (searchTerm.trim()) {
          const q = searchTerm.toLowerCase();
          const matchId = p.zone_id.toLowerCase().includes(q);
          const matchIndicator = p.dominant_indicator.toLowerCase().includes(q);
          const matchAction = p.recommended_action.toLowerCase().includes(q);
          const matchClassName = p.risk_class_name.toLowerCase().includes(q);
          if (!matchId && !matchIndicator && !matchAction && !matchClassName) {
            return false;
          }
        }

        return true;
      })
      .sort((a, b) => {
        let valA = a.properties[sortField];
        let valB = b.properties[sortField];

        if (typeof valA === 'string') {
          valA = (valA as string).toLowerCase();
          valB = ((valB as string) || '').toLowerCase();
        }

        if (valA < valB) return sortDirection === 'asc' ? -1 : 1;
        if (valA > valB) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
  }, [fieldFeatures, activeFilter, searchTerm, sortField, sortDirection, filterByHorizon, daysHorizon]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      // Default to ascending for onset/id, descending for score/area
      setSortDirection(field === 'days_to_onset' || field === 'zone_id' ? 'asc' : 'desc');
    }
  };

  const getRiskBadge = (riskClass: RiskClass) => {
    switch (riskClass) {
      case 0:
        return (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <Leaf className="w-2.5 h-2.5" /> Healthy
          </span>
        );
      case 1:
        return (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-amber-500/15 text-amber-400 border border-amber-500/30">
            <Sparkles className="w-2.5 h-2.5" /> Early Stress
          </span>
        );
      case 2:
        return (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-orange-500/15 text-orange-400 border border-orange-500/30">
            <AlertCircle className="w-2.5 h-2.5" /> High Risk
          </span>
        );
      case 3:
        return (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium bg-rose-500/20 text-rose-400 border border-rose-500/30 animate-pulse">
            <Flame className="w-2.5 h-2.5" /> Visible Blight
          </span>
        );
    }
  };

  const formatIndicator = (indicator: string) => {
    switch (indicator) {
      case 'red_edge_shift':
        return 'Red-Edge Shift (705nm)';
      case 'pri_decline':
        return 'PRI Deficit (531nm)';
      case 'chlorophyll_loss':
        return 'Chlorophyll Decline (680nm)';
      case 'water_stress':
        return 'Water Band (970nm)';
      default:
        return 'Baseline Phenology';
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-200 select-none overflow-hidden">
      {/* Top Header & Distribution Metric Bar */}
      <div className="p-3 border-b border-slate-800/90 bg-slate-900/60 space-y-2.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-bold text-slate-100 uppercase tracking-wider">
              Zone Risk Matrix
            </h3>
            <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
              {processedFeatures.length} of {fieldFeatures.length} zones
            </span>
          </div>

          {stats.urgentCount > 0 && (
            <div className="flex items-center gap-1 text-[10px] font-semibold text-rose-400 bg-rose-500/10 border border-rose-500/30 px-2 py-0.5 rounded-full">
              <AlertTriangle className="w-3 h-3 text-rose-400 animate-pulse" />
              <span>{stats.urgentCount} Critical Urgent</span>
            </div>
          )}
        </div>

        {/* Visual Risk Distribution Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] text-slate-400">
            <span>Canopy Risk Distribution</span>
            <span>{stats.totalAcres} Total Ag Acres</span>
          </div>
          <div className="h-2 w-full bg-slate-800 rounded-full flex overflow-hidden">
            {stats.healthyPct > 0 && (
              <div
                style={{ width: `${stats.healthyPct}%` }}
                className="bg-emerald-500 h-full transition-all"
                title={`Healthy: ${stats.healthyPct}%`}
              />
            )}
            {stats.earlyPct > 0 && (
              <div
                style={{ width: `${stats.earlyPct}%` }}
                className="bg-amber-400 h-full transition-all"
                title={`Early Stress: ${stats.earlyPct}%`}
              />
            )}
            {stats.highBlightPct > 0 && (
              <div
                style={{ width: `${stats.highBlightPct}%` }}
                className="bg-orange-500 h-full transition-all"
                title={`High Blight Risk: ${stats.highBlightPct}%`}
              />
            )}
            {stats.visiblePct > 0 && (
              <div
                style={{ width: `${stats.visiblePct}%` }}
                className="bg-rose-500 h-full transition-all"
                title={`Visible Symptoms: ${stats.visiblePct}%`}
              />
            )}
          </div>
          <div className="flex items-center justify-between text-[9px] text-slate-500 pt-0.5">
            <span className="flex items-center gap-1 text-emerald-400">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block" /> {stats.healthyPct}% Healthy
            </span>
            <span className="flex items-center gap-1 text-amber-400">
              <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" /> {stats.earlyPct}% Early
            </span>
            <span className="flex items-center gap-1 text-orange-400">
              <span className="w-1.5 h-1.5 rounded-full bg-orange-500 inline-block" /> {stats.highBlightPct}% High Risk
            </span>
            {stats.visiblePct > 0 && (
              <span className="flex items-center gap-1 text-rose-400">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-500 inline-block" /> {stats.visiblePct}% Visible
              </span>
            )}
          </div>
        </div>

        {/* Search & Quick Filter Row */}
        <div className="space-y-2">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search by Zone ID, indicator, action..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-slate-950/80 border border-slate-800 rounded-lg pl-8 pr-7 py-1 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                aria-label="Clear search"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>

          {/* Filter Pills */}
          <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5 text-[10px]">
            <button
              onClick={() => setActiveFilter('all')}
              className={`px-2 py-0.5 rounded-md font-medium whitespace-nowrap transition-colors ${
                activeFilter === 'all'
                  ? 'bg-slate-700 text-slate-100 border border-slate-600'
                  : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800/80'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setActiveFilter('urgent')}
              className={`px-2 py-0.5 rounded-md font-medium whitespace-nowrap transition-colors ${
                activeFilter === 'urgent'
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                  : 'bg-slate-900 text-slate-400 hover:text-rose-400 border border-slate-800/80'
              }`}
            >
              Urgent (≤ 7d)
            </button>
            <button
              onClick={() => setActiveFilter('high_risk')}
              className={`px-2 py-0.5 rounded-md font-medium whitespace-nowrap transition-colors ${
                activeFilter === 'high_risk'
                  ? 'bg-orange-500/20 text-orange-300 border border-orange-500/40'
                  : 'bg-slate-900 text-slate-400 hover:text-orange-400 border border-slate-800/80'
              }`}
            >
              High Risk
            </button>
            <button
              onClick={() => setActiveFilter('early_stress')}
              className={`px-2 py-0.5 rounded-md font-medium whitespace-nowrap transition-colors ${
                activeFilter === 'early_stress'
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  : 'bg-slate-900 text-slate-400 hover:text-amber-400 border border-slate-800/80'
              }`}
            >
              Early Stress
            </button>
            <button
              onClick={() => setActiveFilter('healthy')}
              className={`px-2 py-0.5 rounded-md font-medium whitespace-nowrap transition-colors ${
                activeFilter === 'healthy'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                  : 'bg-slate-900 text-slate-400 hover:text-emerald-400 border border-slate-800/80'
              }`}
            >
              Healthy
            </button>
          </div>
        </div>
      </div>

      {/* Sortable Table Header */}
      <div className="grid grid-cols-12 gap-1 px-3 py-2 bg-slate-900/90 border-b border-slate-800 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
        <button
          onClick={() => toggleSort('zone_id')}
          className="col-span-4 flex items-center gap-1 hover:text-slate-200 text-left"
        >
          <span>Zone / Indicator</span>
          <ArrowUpDown className="w-2.5 h-2.5 opacity-60" />
        </button>
        <button
          onClick={() => toggleSort('risk_score')}
          className="col-span-3 flex items-center justify-center gap-1 hover:text-slate-200 text-center"
        >
          <span>Class / Score</span>
          <ArrowUpDown className="w-2.5 h-2.5 opacity-60" />
        </button>
        <button
          onClick={() => toggleSort('area_acres')}
          className="col-span-2 flex items-center justify-end gap-1 hover:text-slate-200 text-right"
        >
          <span>Acres</span>
          <ArrowUpDown className="w-2.5 h-2.5 opacity-60" />
        </button>
        <button
          onClick={() => toggleSort('days_to_onset')}
          className="col-span-3 flex items-center justify-end gap-1 hover:text-slate-200 text-right"
        >
          <span>Onset</span>
          <ArrowUpDown className="w-2.5 h-2.5 opacity-60" />
        </button>
      </div>

      {/* Table Body / Rows */}
      <div className="flex-1 overflow-y-auto divide-y divide-slate-800/60" data-testid="zone-table-body">
        {processedFeatures.length === 0 ? (
          <div className="p-8 text-center text-slate-500 space-y-2">
            <Crosshair className="w-8 h-8 mx-auto text-slate-600 stroke-[1.5]" />
            <p className="text-xs font-semibold text-slate-400">No matching zones found</p>
            <p className="text-[11px] text-slate-500">
              Try adjusting your search query or reset filter options.
            </p>
            <button
              onClick={() => {
                setSearchTerm('');
                setActiveFilter('all');
                setFilterByHorizon(false);
              }}
              className="mt-2 text-xs px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-emerald-400 rounded-md border border-slate-700 transition-colors"
            >
              Reset Filters
            </button>
          </div>
        ) : (
          processedFeatures.map((f) => {
            const p = f.properties;
            const isSelected = selectedZone?.properties.zone_id === p.zone_id;
            const isCriticalOnset = p.risk_class > 0 && p.days_to_onset <= 7;

            return (
              <div
                key={p.zone_id}
                data-testid={`zone-row-${p.zone_id}`}
                onClick={() => onSelectZone(f)}
                className={`grid grid-cols-12 gap-1 px-3 py-2.5 items-center cursor-pointer transition-colors text-xs ${
                  isSelected
                    ? 'bg-slate-800/90 border-l-2 border-emerald-400 shadow-inner'
                    : 'hover:bg-slate-900/80 bg-slate-950/40'
                }`}
              >
                {/* Zone ID & Spectral Indicator */}
                <div className="col-span-4 min-w-0 pr-1">
                  <div className="flex items-center gap-1.5 truncate">
                    <span
                      className={`w-2 h-2 rounded-full shrink-0 ${
                        p.risk_class === 0
                          ? 'bg-emerald-500'
                          : p.risk_class === 1
                          ? 'bg-amber-400'
                          : p.risk_class === 2
                          ? 'bg-orange-500'
                          : 'bg-rose-500 animate-ping'
                      }`}
                    />
                    <span className="font-semibold text-slate-200 truncate">
                      {p.zone_id}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 truncate pl-3.5" title={p.dominant_indicator}>
                    {formatIndicator(p.dominant_indicator)}
                  </div>
                </div>

                {/* Class Badge & Score */}
                <div className="col-span-3 flex flex-col items-center justify-center">
                  {getRiskBadge(p.risk_class)}
                  <span className="text-[10px] text-slate-400 font-mono mt-0.5">
                    {Math.round(p.risk_score * 100)}% score
                  </span>
                </div>

                {/* Area in Acres */}
                <div className="col-span-2 text-right font-mono text-[11px] text-slate-300">
                  {p.area_acres.toFixed(1)} <span className="text-[9px] text-slate-500">ac</span>
                </div>

                {/* Days to Onset Countdown & Action button */}
                <div className="col-span-3 flex items-center justify-end gap-1.5">
                  <div className="text-right">
                    {p.risk_class === 0 ? (
                      <span className="text-[10px] text-emerald-400/80 font-mono">Healthy</span>
                    ) : (
                      <div className="flex flex-col items-end">
                        <span
                          className={`font-mono text-[11px] font-semibold flex items-center gap-1 ${
                            isCriticalOnset ? 'text-rose-400' : 'text-amber-300'
                          }`}
                        >
                          {isCriticalOnset && <Clock className="w-2.5 h-2.5 text-rose-400" />}
                          {p.days_to_onset === 0 ? 'Now' : `${p.days_to_onset.toFixed(0)}d`}
                        </span>
                        <span className="text-[9px] text-slate-500">
                          {p.days_to_onset === 0 ? 'symptomatic' : 'pre-visual'}
                        </span>
                      </div>
                    )}
                  </div>

                  {onInspectZone && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectZone(f);
                        onInspectZone(f);
                      }}
                      className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-emerald-400 transition-colors"
                      title="Inspect Agronomic Prescription"
                      aria-label={`Inspect ${p.zone_id}`}
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Quick Footer Summary */}
      <div className="p-2 border-t border-slate-800 bg-slate-900/80 px-3 flex items-center justify-between text-[11px] text-slate-400">
        <span>Click row to focus on Map</span>
        {selectedZone && (
          <span className="text-emerald-400 font-medium flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            Selected: {selectedZone.properties.zone_id}
          </span>
        )}
      </div>
    </div>
  );
};
