"""Synthetic Track B engineering regressions; no research solver or training."""
from __future__ import annotations
import copy
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from track_b.cape_executor import OriginalDomainCapeExecutor, SharedWitnessCompleter, quantize_flight_ms
from track_b.independent_comparator import compare_actual_logs_against_expected
from track_b.run_track_b_acceptance import run_blind_executor_all_traces


def spec():
    return {'trace_id':'SYNTHETIC', 'initial_state':{
        't_start_ms':0, 'shift_end_ms':600000, 'budget_total_yuan':'20',
        'stations':[{'station_id':'S', 'station_type':'SABS', 'x_m':0, 'y_m':0}],
        'tower_catalog':[{'node_id':'n', 'x_m':300, 'y_m':0, 'default_routine_srv_ms':60000}],
        'mandatory_routine_tasks':[{'task_id':'M', 'station_id':'S', 'node_id':'n', 'srv_ms':60000}],
        'drones':[{'drone_id':'D', 'station_id':'S', 'slot_idx':1, 'model':'SD', 'initial_epoch':0,
                   'initial_witness_ver':'w', 'initial_witness_sorties':[
                       {'sortie_id':'R', 't_arm_ms':100000, 'stops':[{'node_id':'n','srv_ms':60000},{'node_id':'S','srv_ms':0}]}]}]},
        'exogenous_events':[]}


def proposal(sim, stops=None):
    return {'proposal_id':'P', 'proposal_seq':1, 'station_id':'S', 'drone_id':'D',
            'snapshot_epoch':sim.drones['D']['epoch'], 'snapshot_witness_ver':sim.drones['D']['witness_ver'],
            'new_witness_ver':'w2', 'candidate_sorties':[{'sortie_id':'C','t_arm_ms':100000,
            'stops':stops or [{'node_id':'n','srv_ms':60000},{'node_id':'S','srv_ms':0}]}]}


class TestTrackBReview(unittest.TestCase):
    def frozen(self):
        raw=json.loads((ROOT/'track_b/inputs/track_b_v5_1_inputs.json').read_text())
        expected=json.loads((ROOT/'track_b/expected/track_b_v5_1_expected.json').read_text())
        actual={'traces_actual_logs':{tr['trace_id']:OriginalDomainCapeExecutor(tr).run_to_shift_end() for tr in raw['traces']}}
        return actual,expected

    def test_original_37_checkpoints_unchanged(self):
        actual,expected=self.frozen()
        report=compare_actual_logs_against_expected(actual,expected)
        self.assertTrue(report['all_passed'],report)
        self.assertEqual(report['total_verified_checkpoints'],37)

    def test_nested_expected_rejected(self):
        s=spec(); s['initial_state']['metadata']={'expected_verdict':'PASS'}
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_duplicate_drone_rejected(self):
        s=spec(); s['initial_state']['drones'].append(copy.deepcopy(s['initial_state']['drones'][0]))
        s['initial_state']['drones'][1]['slot_idx']=-1
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_negative_slot_rejected(self):
        s=spec(); s['initial_state']['drones'][0]['slot_idx']=-1
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_initial_witness_must_close(self):
        s=spec(); s['initial_state']['drones'][0]['initial_witness_sorties'][0]['stops'].pop()
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_initial_battery_and_shift_hard_guards(self):
        for field,value in [('shift_end_ms',1000),('budget_total_yuan','0')]:
            s=spec();s['initial_state'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)
        s=spec(); s['initial_state']['mandatory_routine_tasks'][0]['srv_ms']=3000000
        s['initial_state']['drones'][0]['initial_witness_sorties'][0]['stops'][0]['srv_ms']=3000000
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_invalid_event_time_and_type_rejected(self):
        for event in [{'t_ms':600001,'type':'PUBLISH_TICKET','ticket':{'ticket_id':'e','node_id':'n','sla_ms':600000,'srv_ms':0}},
                      {'t_ms':-1,'type':'UNKNOWN'}, {'t_ms':10,'type':'UNKNOWN'}]:
            s=spec();s['exogenous_events']=[event]
            with self.subTest(event=event),self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_duplicate_ticket_publication_rejected(self):
        event={'t_ms':1,'type':'PUBLISH_TICKET','ticket':{'ticket_id':'e','node_id':'n','sla_ms':600000,'srv_ms':0}}
        s=spec();s['exogenous_events']=[event,copy.deepcopy(event)]
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_empty_candidate_station_mismatch_and_duplicate_ticket(self):
        for case in ['empty','station','duplicate','negative','return_ticket']:
            sim=OriginalDomainCapeExecutor(spec())
            sim.tickets['e']={'ticket_id':'e','node_id':'n','sla_ms':600000,'srv_ms':10000,'status':'UNCLAIMED'}
            p=proposal(sim)
            if case=='empty':p['candidate_sorties']=[]
            if case=='station':p['station_id']='OTHER'
            if case=='negative':p['candidate_sorties'][0]['stops'][0]['srv_ms']=-1
            if case=='duplicate':p['candidate_sorties'][0]['stops'][1:1]=[{'node_id':'n','srv_ms':10000,'ticket_id':'e'}]*2
            if case=='return_ticket':p['candidate_sorties'][0]['stops'][-1]['ticket_id']='e'
            with self.subTest(case=case):
                result=sim.atomic_cas(p)
                self.assertTrue(result['verdict'].startswith('REJECT_'),result)
                self.assertEqual(sim.ledger_ver,0)
                self.assertEqual(sim.tickets['e']['status'],'UNCLAIMED')

    def test_multiple_routine_tasks_same_tower_keep_identity(self):
        s=spec();init=s['initial_state']
        init['mandatory_routine_tasks'].append({'task_id':'M2','station_id':'S','node_id':'n','srv_ms':20000})
        stops=init['drones'][0]['initial_witness_sorties'][0]['stops']
        stops[0]['task_id']='M'
        stops.insert(1,{'node_id':'n','srv_ms':20000,'task_id':'M2'})
        sim=OriginalDomainCapeExecutor(s)
        self.assertEqual(sim.drones['D']['unserved_routine'],{'M','M2'})
        p=proposal(sim,[{'node_id':'n','srv_ms':60000,'task_id':'M'},{'node_id':'S','srv_ms':0}])
        self.assertEqual(sim.atomic_cas(p)['verdict'],'REJECT_COMMITTED_WORK_DROPPED')
        logs=sim.run_to_shift_end()
        self.assertTrue(logs[-1]['mandatory_routine_all_done'])
        self.assertEqual(logs[-1]['mandatory_routine_completed_count'],2)

    def test_duplicate_initial_assignment_rejected(self):
        s=spec();init=s['initial_state'];init['stations'][0]['station_type']='DABS'
        d=copy.deepcopy(init['drones'][0]);d.update(drone_id='D2',slot_idx=0,model='LD')
        init['drones'].append(d)
        with self.assertRaises(ValueError): OriginalDomainCapeExecutor(s)

    def test_comparator_rejects_extra_trace_empty_expected_and_negative_raw_bucket(self):
        actual,expected=self.frozen()
        cases=[]
        extra=copy.deepcopy(actual);extra['traces_actual_logs']['EXTRA']=next(iter(extra['traces_actual_logs'].values()))
        cases.append((extra,expected))
        cases.append((actual,{'expected_traces':{}}))
        negative=copy.deepcopy(actual)
        lg=next(lg for logs in negative['traces_actual_logs'].values() for lg in logs if lg['event_type']=='DEFAULT_TAKEOFF_FIRED')
        from fractions import Fraction
        total=sum(map(Fraction,lg['buckets_exact']))
        lg['buckets_exact']=['-1',str(total+1),'0','0']
        cases.append((negative,expected))
        for a,e in cases:
            with self.subTest():self.assertFalse(compare_actual_logs_against_expected(a,e)['all_passed'])

    def test_comparator_rejects_malformed_logs_without_crashing(self):
        actual,expected=self.frozen()
        for replacement in [None,{},[],[{'t_ms':0}], [{'event_type':'SHIFT_INIT','t_ms':0,'buckets_exact':['x']}]]:
            a=copy.deepcopy(actual);a['traces_actual_logs'][next(iter(a['traces_actual_logs']))]=replacement
            with self.subTest(replacement=replacement):self.assertFalse(compare_actual_logs_against_expected(a,expected)['all_passed'])

    def test_comparator_checks_noncheckpoint_hard_violation_and_time(self):
        actual,expected=self.frozen()
        for case in ['hard','time','shape']:
            a=copy.deepcopy(actual);logs=next(iter(a['traces_actual_logs'].values()))
            lg=next(lg for lg in logs if lg['event_type']=='DEFAULT_TAKEOFF_FIRED')
            if case=='hard':lg['hard_violations']=1
            if case=='time':lg['t_ms']=-1
            if case=='shape':lg['buckets_exact'].append('0')
            with self.subTest(case=case):self.assertFalse(compare_actual_logs_against_expected(a,expected)['all_passed'])

    def test_blind_runner_rejects_duplicate_ids_and_nested_expected(self):
        for case in ['duplicate','expected']:
            raw={'traces':[spec()]}
            if case=='duplicate':raw['traces'].append(spec())
            else:raw['expected_checkpoints']=[]
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'in.json';p.write_text(json.dumps(raw))
                with self.subTest(case=case),self.assertRaises(ValueError):run_blind_executor_all_traces(p)

    def test_write_expected_backdoor_removed(self):
        with tempfile.TemporaryDirectory() as td:
            args=[sys.executable,str(ROOT/'track_b/run_track_b_acceptance.py'),'--write-expected','--expected',str(Path(td)/'expected.json')]
            for flag in ['--b3-trace-out','--actual-logs-out','--timing-out','--comparison-out']:
                args.extend([flag,str(Path(td)/(flag[2:]+'.json'))])
            result=subprocess.run(args,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('unrecognized arguments',result.stderr)

    def test_selector_cannot_replace_completed_witness(self):
        sim=OriginalDomainCapeExecutor(spec())
        def forged(_snapshot, feasible):
            for act,_comp in feasible:
                if act[0]=='STEP':return act,[{'sortie_id':'FORGED','t_arm_ms':0,'stops':[{'node_id':'n','srv_ms':60000},{'node_id':'S','srv_ms':0}]}]
            self.fail('No physical action')
        with self.assertRaises(ValueError):SharedWitnessCompleter.execute_timed_decision_step(sim,forged,'P',1)
        self.assertEqual(sim.ledger_ver,0)

    def test_decision_wall_time_advances_physics_before_cas(self):
        s=spec();s['initial_state']['drones'][0]['initial_witness_sorties'][0]['t_arm_ms']=2
        sim=OriginalDomainCapeExecutor(s)
        def delayed(_snapshot,feasible):
            act=next(act for act,_comp in feasible if act[0]=='STEP')
            time.sleep(.006)
            return act
        result=SharedWitnessCompleter.execute_timed_decision_step(sim,delayed,'P',1)
        self.assertGreaterEqual(sim.t_ms,6)
        self.assertEqual(result['cas_verdict'],'REJECT_DRONE_IN_FLIGHT_NO_DIVERSION')
        self.assertGreaterEqual(result['wall_decision_to_commit_ms'],6)

    def test_completer_bounded_limit_is_not_infeasibility(self):
        s=spec();init=s['initial_state'];init['mandatory_routine_tasks']=[]
        stops=[]
        for i in range(7):
            init['tower_catalog'].append({'node_id':f'n{i}','x_m':300+i,'y_m':0})
            init['mandatory_routine_tasks'].append({'task_id':f'M{i}','station_id':'S','node_id':f'n{i}','srv_ms':1})
            stops.append({'node_id':f'n{i}','srv_ms':1})
        init['drones'][0]['initial_witness_sorties'][0]['stops']=stops+[{'node_id':'S','srv_ms':0}]
        sim=OriginalDomainCapeExecutor(s)
        _comp,status=SharedWitnessCompleter.complete_after_explicit_next_step(sim,'S',1,('ROUTINE','n0'))
        self.assertEqual(status,'COMPLETION_LIMIT_EXCEEDED')

    def test_settle_rejects_backward_time(self):
        sim=OriginalDomainCapeExecutor(spec());sim._settle_to(1)
        with self.assertRaises(ValueError):sim._settle_to(0)

    def test_completer_searches_intermediate_sortie_partitions(self):
        s=spec();init=s['initial_state'];init['shift_end_ms']=6000000
        init['mandatory_routine_tasks']=[];init['tower_catalog']=[]
        sorties=[]
        for i in range(3):
            init['tower_catalog'].append({'node_id':f'n{i}','x_m':300+i,'y_m':0})
            init['mandatory_routine_tasks'].append({'task_id':f'M{i}','station_id':'S','node_id':f'n{i}','srv_ms':1500000})
            sorties.append({'sortie_id':f'R{i}','t_arm_ms':100000,'stops':[{'node_id':f'n{i}','srv_ms':1500000},{'node_id':'S','srv_ms':0}]})
        init['drones'][0]['initial_witness_sorties']=sorties
        sim=OriginalDomainCapeExecutor(s)
        completion,status=SharedWitnessCompleter.complete_after_explicit_next_step(sim,'S',1,('ROUTINE','n0'))
        self.assertEqual(status,'OK')
        self.assertEqual(len(completion),3)
        self.assertEqual(completion[0]['stops'][0]['node_id'],'n0')
        self.assertIsNone(sim.evaluate_candidate_rollout_metrics(sim.drones['D'],completion)['batt_over_sortie_idx'])

    def test_flight_ceiling_never_snaps_fractional_coordinate_down(self):
        self.assertEqual(quantize_flight_ms((0,0),(.015+1e-15,0)),2)
        self.assertEqual(quantize_flight_ms((0,0),(3+1e-10,0)),201)

    def test_runner_blind_only_never_reads_expected(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td);inputs=folder/'inputs.json';inputs.write_text(json.dumps({'traces':[spec()]}))
            args=[sys.executable,str(ROOT/'track_b/run_track_b_acceptance.py'),'--blind-only','--inputs',str(inputs),'--expected',str(folder/'does-not-exist.json')]
            for flag in ['--b3-trace-out','--actual-logs-out','--timing-out','--comparison-out']:
                args.extend([flag,str(folder/(flag[2:]+'.json'))])
            result=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['comparison_performed'],False)
            self.assertFalse((folder/'comparison-out.json').exists())
            self.assertEqual(json.loads((folder/'actual-logs-out.json').read_text())['evidence_scope'],'SYNTHETIC_ENGINEERING_REPLAY_ONLY_NOT_RESEARCH_OR_RL_RESULTS')

    def test_output_paths_cannot_overwrite_expected(self):
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'expected.json';target.write_text('PRESERVE')
            result=subprocess.run([sys.executable,str(ROOT/'track_b/run_track_b_acceptance.py'),'--expected',str(target),'--actual-logs-out',str(target)],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertEqual(target.read_text(),'PRESERVE')

    def test_budget_guards_survive_python_optimized_mode(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'input.json';p.write_text(json.dumps(spec()))
            code="import json,sys;from track_b.cape_executor import OriginalDomainCapeExecutor;s=OriginalDomainCapeExecutor(json.load(open(sys.argv[1])));s.B_spent=-1;s.B_free+=1;s._record('INVALID')"
            result=subprocess.run([sys.executable,'-O','-c',code,str(p)],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Budget invariant violated',result.stderr)

    def test_claimed_ticket_stays_selectable_without_duplicate_service(self):
        sim=OriginalDomainCapeExecutor(spec())
        sim.tickets['e']={'ticket_id':'e','node_id':'n','sla_ms':600000,'srv_ms':10000,'status':'UNCLAIMED'}
        p=proposal(sim,[{'node_id':'n','srv_ms':60000},{'node_id':'n','srv_ms':10000,'ticket_id':'e'},{'node_id':'S','srv_ms':0}])
        self.assertTrue(sim.atomic_cas(p)['verdict'].startswith('PASS_'))
        actions=SharedWitnessCompleter.enumerate_quotient_physical_actions(sim)
        self.assertIn(('STEP','S',1,('TICKET','e','n')),actions)
        completion,status=SharedWitnessCompleter.complete_after_explicit_next_step(sim,'S',1,('TICKET','e','n'))
        self.assertEqual(status,'OK')
        self.assertEqual(sum(stop.get('ticket_id')=='e' for sortie in completion for stop in sortie['stops']),1)


    def test_policy_exposes_only_physical_steps_and_no_wait(self):
        sim=OriginalDomainCapeExecutor(spec())
        self.assertTrue(all(action[0]=='STEP' for action in SharedWitnessCompleter.enumerate_quotient_physical_actions(sim)))
        sim.advance_to(110000)
        self.assertEqual(SharedWitnessCompleter.enumerate_quotient_physical_actions(sim),[])
        def selector_should_not_run(_snapshot,_feasible):
            self.fail('A policy cannot select WAIT when no physical action exists')
        result=SharedWitnessCompleter.execute_timed_decision_step(sim,selector_should_not_run,'P',1)
        self.assertEqual(result['cas_verdict'],'NO_EXECUTOR_ACTION')
        self.assertIsNone(result['chosen_action'])
        self.assertEqual(result['feasible_action_count'],0)



if __name__=='__main__': unittest.main()
