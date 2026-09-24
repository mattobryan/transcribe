from src.transcribe.pilot.batch import pair_files


def test_pairs_by_normalised_name(tmp_path):
    for name in ["NRCCW_ KSM09.MP3", "NRCCW_KSM09.docx", "NRCCW_KSM09.pdf",
                 "NRCCW_KSM10.wav", "Other.docx", "~$NRCCW_KSM09.docx"]:
        (tmp_path / name).write_bytes(b"x")
    pairs, lonely_text, lonely_audio = pair_files(tmp_path)
    assert [(a.name, t.name) for a, t in pairs] == [("NRCCW_ KSM09.MP3", "NRCCW_KSM09.docx")]
    assert [p.name for p in lonely_text] == ["Other.docx"]
    assert [p.name for p in lonely_audio] == ["NRCCW_KSM10.wav"]
