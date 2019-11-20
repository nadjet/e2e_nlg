from e2e_nlg.evaluation.mr_formatter import MR_Formatter
from e2e_nlg.evaluation.relexicalizer import ReLexicalizer
from e2e_nlg.evaluation.ref_formatter import ReferenceFormatter
from e2e_nlg.evaluation.out_formatter import OutputFormatter
import os
from utils.log import logger


class DataFormattingPipeline:
    def __init__(self,output_file=None,ref_file=None,output_folder=None):
        self.output_file = output_file
        self.ref_file = ref_file
        self.output_folder = output_folder

    def prepare_data(self):
        logger.info("Formatting the MRs in the output...")
        formatted_file = os.path.join(self.output_folder,"formatted_output.csv")
        formatter = MR_Formatter(self.output_file, formatted_file)
        formatter.process_mr()
        logger.info("...Formatting the MRs in the output done! Written to: {}".format(formatted_file))

        logger.info("Relexicalizing output MRs and texts...")
        lexicalized_file = os.path.join(self.output_folder,"lexicalized_output.csv")
        relexicalizer = ReLexicalizer(formatted_file, self.ref_file, lexicalized_file)
        relexicalizer.relexicalize()
        logger.info("...Relexicalizing output MRs and texts done! Written to: {}".format(lexicalized_file))

        logger.info("Splitting reference into MRs and output files...")
        ref_formatter = ReferenceFormatter(self.ref_file, self.output_folder)
        ref_formatter.set_mr_groups_and_mrs()
        refs_file = ref_formatter.save_ref_groups()
        mrs_file = ref_formatter.save_mrs()
        logger.info("...Splitting reference into MRs and output files done!")

        logger.info("Writing output in same order as MRs...")
        out_formatter = OutputFormatter(mrs_file, lexicalized_file, self.output_folder)
        preds_file = out_formatter.write_predictions()
        logger.info("...Writing output in same order as MRs done!")

        logger.info("References file={}".format(refs_file))
        logger.info("Outputs file={}".format(preds_file))


